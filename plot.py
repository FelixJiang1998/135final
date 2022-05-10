import numpy as np
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot import plt
import seaborn as sns

def visualization(X, y_target):

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000,random_state=135)
    low_dim_embs = tsne.fit_transform(X)
    labels = y_target.ravel()

    X, Y = low_dim_embs[:, 0], low_dim_embs[:, 1]

    classes = list(np.unique(labels))
    markers = 'os' * len(classes)
    colors = sns(np.linspace(0, 1, len(classes)))
    for x, y, s in zip(X, Y, labels):
        i = int(s)
        plt.scatter(x, y, marker=markers[i], c=[colors[i]], alpha=0.3)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')

    plt.legend()
    plt.axis("off")

    plt.show()


def save_model():
    checkpoint = {
        "net": cnn.state_dict(),
        'optimizer':optimizer.state_dict(),
        "epoch": epoch
    }

    fomat_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    torch.save(checkpoint, 'models/checkpoint/autosave_'+fomat_time)