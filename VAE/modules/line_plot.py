import matplotlib.pyplot as plt


class LinePlot:
    def __init__(self):
        pass

    @staticmethod
    def learning_curve(x, y, title=None):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Epoch')

        if isinstance(y, dict):
            for label, y_val in y.items():
                ax.plot(x, y_val, label=label)
            ax.legend()
        else:
            ax.plot(x, y)
        ax.set_title(title)
        fig.savefig(title, facecolor='w')
        plt.show()
