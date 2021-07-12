import matplotlib.pyplot as plt
from torchvision import transforms


class ImagePlot:
    def __init__(self):
        pass
    
    @staticmethod
    def sample_images(images_tensor, shape, title, wspace=-0.85, hspace=0.01):
        #img_num_each_edge = int(np.ceil((len(images_tensor))**0.5))
        fig = plt.figure()
        for i in range(len(images_tensor)):
            ax = fig.add_subplot(shape[0], shape[1], i+1)
            img = transforms.ToPILImage()(images_tensor[i])
            ax.imshow(img)
            ax.axis('off')

        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        fig.savefig(title, facecolor='w')
        plt.show()
