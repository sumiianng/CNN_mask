import matplotlib.pyplot as plt


class line_plot:
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
        plt.show()

    @staticmethod
    def hparams_learning_curve(rundata_df, y, label, title=None):
        if isinstance(label, tuple):
            label = list(label)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel('Epoch')
        for label_values, df in rundata_df.groupby(label):
            x = df['epoch']
            y_val = df[y]
            label_str = [f"{i}={j}" for i, j in zip(label, label_values)]
            label_str = ",".join(label_str)
            ax.plot(x, y_val, label=label_str)

        ax.set_title(title)
        ax.legend()
        plt.show()
