import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision.datasets import DatasetFolder
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.manifold import TSNE

torch.manual_seed(135)


class NeuroNetwork(nn.Module):
    def __init__(self):
        super(NeuroNetwork, self).__init__()
        first_layer = 64
        second_layer = 128
        third_layer = 256
        forth_layer = 512
        fc_layer = 1024
        output = 25
        self.cnn_layers = nn.Sequential(
            # First layer
            nn.Conv2d(in_channels=3,  # RGB 3 layer3
                      out_channels=first_layer,  # Output layer -- the number of filters
                      kernel_size=3,  # Size of filter --3*3
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(first_layer),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Second layer
            nn.Conv2d(first_layer, second_layer, 3, 1, 1),
            nn.BatchNorm2d(second_layer),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third layer
            nn.Conv2d(second_layer, third_layer, 3, 1, 1),
            nn.BatchNorm2d(third_layer),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Forth layer
            nn.Conv2d(third_layer, forth_layer, 3, 1, 1),
            nn.BatchNorm2d(forth_layer),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(forth_layer * 8 * 8, fc_layer),
            nn.BatchNorm1d(fc_layer),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_layer, output)
        )

    def forward(self, x):
        x = self.cnn_layers(x)

        x = x.flatten(1)
        visual_item = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x, visual_item


train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
    transforms.RandomRotation(degrees=25),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Data
train_set = DatasetFolder("dataset/arcDataset", loader=lambda x: Image.open(x).convert("RGB"), extensions="jpg",
                          transform=train_tfm)
test_set = DatasetFolder("dataset/arcValidset", loader=lambda x: Image.open(x).convert("RGB"), extensions="jpg",
                         transform=test_tfm)

# Batch size of 128
batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"

# CrossEntropy loss are applied
cross_entropy = nn.CrossEntropyLoss()

cnn = NeuroNetwork().to(device)
cnn.device = device
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-5)

train_loss_record = []
valid_loss_record = []
train_acc_record = []
valid_acc_record = []


def save_model():
    checkpoint = {
        "net": cnn.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch
    }

    fomat_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    torch.save(checkpoint, 'models/checkpoint/autosave_' + fomat_time)


def plot_with_labels(lowDWeights, labels):
    # plt.cla()
    # fig, ax = plt.subplots()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    labels = np.ravel(labels)
    classes = list(np.unique(labels))
    markers = 'os' * len(classes)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    for x, y, s in zip(X, Y, labels):
        i = int(s)
        # plt.text(x, y, s, backgroundcolor=colors[i], fontsize=8)
        plt.scatter(x, y, marker=markers[i], c=[colors[i]], alpha=0.3)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')

    plt.legend()
    plt.axis("off")
    # fig.set_facecolor('k')
    plt.show()


# Train for 20 times rounds
n_epochs = 150
best_acc = 0.0
for epoch in range(n_epochs):
    print("Epoch: ", epoch)
    start_time = time.time()

    cnn.train()

    train_loss = []
    train_acc = []

    for batch in train_loader:
        data, labels = batch

        predict, _ = cnn(data.to(device))
        loss = cross_entropy(predict, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = torch.tensor(predict.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item())
        train_acc.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    cnn.eval()

    test_loss = []
    test_acc = []
    visual_label_set = []
    visual_predict_set = []
    for batch in test_loader:
        data, labels = batch

        with torch.no_grad():
            predict, last_layer_item = cnn(data.to(device))

        loss = cross_entropy(predict, labels.to(device))

        acc = torch.tensor(predict.argmax(dim=-1) == labels.to(device)).float().mean()

        test_loss.append(loss.item())
        test_acc.append(acc)
        visual_predict_set.append(last_layer_item)
        visual_label_set.append(labels.to(device).unsqueeze(-1))
    print("squeezing data")
    v_last_layer_item = torch.vstack(visual_predict_set)
    v_labels = torch.vstack(visual_label_set)

    print("visualizing")
    # Visualization of trained flatten layer (T-SNE)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(v_last_layer_item.data.numpy()[:, :])
    labels = v_labels.to(device).numpy()[:]
    plot_with_labels(low_dim_embs, labels)

    valid_loss = sum(test_loss) / len(test_loss)
    valid_acc = sum(test_acc) / len(test_acc)

    print(f"[ Test | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    if valid_acc > best_acc:
        best_acc = valid_acc

    train_loss_record.append(train_loss)
    valid_loss_record.append(valid_loss)
    train_acc_record.append(train_acc)
    valid_acc_record.append(valid_acc)

    end_time = time.time()

    print(f"[Time cost | {epoch + 1:03d}/{n_epochs:03d}]:{end_time - start_time: .4f}s")

x = np.arange(len(train_acc_record))
plt.plot(x, train_acc_record, color="blue", label="Train")
plt.plot(x, valid_acc_record, color="red", label="Valid")
plt.legend(loc="upper right")
plt.show()

x = np.arange(len(train_loss_record))
plt.plot(x, train_loss_record, color="blue", label="Train")
plt.plot(x, valid_loss_record, color="red", label="Valid")
plt.legend(loc="upper right")
plt.show()
