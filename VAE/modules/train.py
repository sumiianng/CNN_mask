import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict

from modules.run_manager import RunManager
from modules.run_builder import RunBuilder
from modules.vae_network import VAE


def train(train_set, KL_coefficient, device):
    # constant hyper-parameter
    epochs_each_run = 1500

    # variable hyper-parameter
    params = OrderedDict(
        lr=[.001],
        batch_size=[100],
    )

    m = RunManager()
    criterion = nn.MSELoss(reduction='sum')
    for run in RunBuilder.get_runs(params):
        network = VAE().to(device)
        loader = DataLoader(train_set, batch_size=run.batch_size)
        optimizer = optim.Adam(network.parameters(), lr=run.lr)

        m.begin_run(run, network, loader)
        for epoch in range(epochs_each_run):
            m.begin_epoch()
            for batch in loader:
                images = batch.to(device)

                images_re, mu, ln_var = network(images)
                re_loss = criterion(images_re, images)
                kl_div = - 0.5 * torch.sum(1 + ln_var - mu.pow(2) - ln_var.exp())
                loss = re_loss + KL_coefficient*kl_div

                optimizer.zero_grad()
                loss.backward()  # Gradients
                optimizer.step()  # Update Weights

                m.track_loss(loss)
            m.end_epoch(100)
        m.end_run()

    return network, m
