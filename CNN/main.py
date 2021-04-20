import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from collections import OrderedDict

from modules.network import Network
from modules.run_builder import RunBuilder
from modules.preprocess import Preprocess
from modules.dataset import MaskDataset
from modules.run_manager import RunManager
from modules.line_plot import line_plot
from modules.confusion_matrix import ConfusionMatrix
from modules.show_label import show_label_on_image

torch.set_grad_enabled(True)


# preporcess data
Preprocess(root='.', train=True, download=False, modify="not_modify")
Preprocess(root='.', train=False, download=False, modify="not_modify")

# dataset
train_set = MaskDataset(
    root='.',
    train=True,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)


test_set = MaskDataset(
    root='.',
    train=False,
    transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

# network
network = Network


# train model
# set device
print(f"Gpu available : {torch.cuda.is_available()}")
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# constant hyper-parameter
num_workers = 0
epochs_each_run = 200

# variable hyper-parameter
params = OrderedDict(
    lr=[.007],
    batch_size=[500],
    # num_workers=[0],
    # device=['cuda'],
)

m = RunManager()
for run in RunBuilder.get_runs(params):
    print(run)
    # device = torch.device(run.device)
    network = Network().to(device)
    loader = DataLoader(train_set, batch_size=run.batch_size)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    m.begin_run(run, network, loader)
    for epoch in range(epochs_each_run):
        m.begin_epoch()
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()  # Gradients
            optimizer.step()  # Update Weights

            m.track_loss(loss, batch)
            m.track_num_correct(preds, labels)

        with torch.no_grad():
            test_loader = DataLoader(test_set,
                                     batch_size=len(test_set.targets))
            for test_batch in test_loader:
                test_imgs = test_batch[0].to(device)
                test_labels = test_batch[1].to(device)
                test_preds = network(test_imgs)
                test_loss = F.cross_entropy(test_preds, test_labels)

                m.track_test_loss(test_loss, test_imgs)
                m.track_test_accuracy(test_preds, test_labels)
        m.end_epoch()
    m.end_run()

pd.DataFrame(m.run_data).sort_values('train_acc', ascending=False).head(20)

# line_plot
df_all = pd.DataFrame(m.run_data)

ys = {'train': df_all['train_acc'], 'test': df_all['test_acc']}
line_plot.learning_curve(df_all['epoch'], ys, title="Accuracy")

ys = {'train': df_all['train_loss'], 'test': df_all['test_loss']}
line_plot.learning_curve(df_all['epoch'], ys, title="Cross entropy")

line_plot.hparams_learning_curve(df_all, y='train_acc',
                                 label=('lr','batch_size'), title="Accuracy_train")

line_plot.hparams_learning_curve(df_all, y='train_loss',
                                 label=('lr','batch_size'), title="Cross_entropy_train")

# confusion matrix
with torch.no_grad():
    train_loader = DataLoader(train_set, batch_size=1000)
    train_preds = ConfusionMatrix.get_all_preds(network, train_loader)

    test_loader = DataLoader(test_set, batch_size=1000)
    test_preds = ConfusionMatrix.get_all_preds(network, test_loader)

train_cm = ConfusionMatrix.get_confusion_matrix(
    train_set.targets, train_preds, len(train_set.classes))
test_cm = ConfusionMatrix.get_confusion_matrix(
    test_set.targets, test_preds, len(test_set.classes))

ConfusionMatrix.plot_confusion_matrix(train_cm,
                                      train_set.classes,
                                      title='Train confusion matrix')
ConfusionMatrix.plot_confusion_matrix(test_cm,
                                      test_set.classes,
                                      title='Test confusion matrix')

cm_dict = {'Train': train_cm, 'Test': test_cm}
classes_acc_df = ConfusionMatrix.get_classes_acc(cm_dict, train_set.classes)
classes_acc_df.round(2)


# samploe result
with torch.no_grad():
    train_loader = DataLoader(train_set, batch_size=1000)
    train_preds = ConfusionMatrix.get_all_preds(network.to('cpu'), train_loader)

show_label_on_image(root=".", all_preds=train_preds, train=True,
                    filename='5db917bb89170.jpg', fontsize=70)
