import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from itertools import product
# from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
    def __init__():
        pass

    @staticmethod
    @torch.no_grad()
    def get_all_preds(model, loader):
        all_preds = torch.tensor([])
        for batch in loader:
            images, labels = batch

            preds = model(images)
            all_preds = torch.cat(
                (all_preds, preds),
                dim=0
            )
        return all_preds

    @staticmethod
    def get_confusion_matrix(targets, preds, class_num):
        # cm = confusion_matrix(targets, preds.argmax(dim=1))
        # return cm

        stacked = torch.stack((targets,
                               preds.argmax(axis=1)), dim=1)

        cmt = torch.zeros(class_num, class_num, dtype=torch.int64)
        for p in stacked:
            tl, pl = p.tolist()
            cmt[tl, pl] = cmt[tl, pl] + 1
            cmt

        return cmt

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False,
                              title='Confusion matrix', cmap=plt.cm.Blues):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ma_plot = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        fig.colorbar(ma_plot, ax=ax)

        ax.xaxis.set_tick_params(rotation=45)
        ax.yaxis.set_tick_params()
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        ax.tick_params(bottom=False, left=False)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        fig.tight_layout()

        plt.show()

    def get_classes_acc(cm_dict, classes):
        result = {}
        for col, cm in cm_dict.items():
            result[col] = torch.diag(cm)/cm.sum(dim=1)
        df = pd.DataFrame(result, index=classes)

        return df
