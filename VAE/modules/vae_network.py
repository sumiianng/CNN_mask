import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, image_size=64, h_dim=128, z_dim=20):
        super().__init__()

        self.ec_conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                                  kernel_size=6, stride=2)
        self.ec_bn1 = nn.BatchNorm2d(6)
        self.ec_conv2 = nn.Conv2d(in_channels=6, out_channels=12,
                                  kernel_size=6, stride=2)
        self.ec_bn2 = nn.BatchNorm2d(12)
        self.ec_conv3 = nn.Conv2d(in_channels=12, out_channels=24,
                                  kernel_size=5, stride=2)
        self.ec_bn3 = nn.BatchNorm2d(24)

        self.ec_fc1 = nn.Linear(24*5*5, h_dim)
        self.ec_bn4 = nn.BatchNorm1d(h_dim)
        self.ec_fc2_mu = nn.Linear(h_dim, z_dim)
        self.ec_fc2_lnvar = nn.Linear(h_dim, z_dim)

        self.dc_fc1 = nn.Linear(z_dim, h_dim)
        self.dc_bn1 = nn.BatchNorm1d(h_dim)
        self.dc_fc2 = nn.Linear(h_dim, 24*5*5)
        self.dc_bn2 = nn.BatchNorm1d(24*5*5)
        self.dc_conv1 = nn.ConvTranspose2d(in_channels=24, out_channels=12,
                                           kernel_size=5, stride=2)
        self.dc_bn3 = nn.BatchNorm2d(12)
        self.dc_conv2 = nn.ConvTranspose2d(in_channels=12, out_channels=6,
                                           kernel_size=6, stride=2)
        self.dc_bn4 = nn.BatchNorm2d(6)
        self.dc_conv3 = nn.ConvTranspose2d(in_channels=6, out_channels=3,
                                           kernel_size=6, stride=2)
        self.dc_bn5 = nn.BatchNorm2d(3)

    def encode(self, t):
        t = self.ec_conv1(t)
        t = F.relu(self.ec_bn1(t))
        t = self.ec_conv2(t)
        t = F.relu(self.ec_bn2(t))
        t = self.ec_conv3(t)
        t = F.relu(self.ec_bn3(t))
        t = self.ec_fc1(t.reshape(-1, 24*5*5))
        t = F.relu(self.ec_bn4(t))

        mu = self.ec_fc2_mu(t)
        ln_var = self.ec_fc2_lnvar(t)

        return mu, ln_var

    def reparameter(self, mu, ln_var):
        std = torch.exp(ln_var / 2)
        epsilon = torch.randn_like(ln_var)

        return mu + std*epsilon

    def decode(self, t):
        t = self.dc_fc1(t)
        t = F.relu(self.dc_bn1(t))
        t = self.dc_fc2(t)
        t = F.relu(self.dc_bn2(t))
        t = self.dc_conv1(t.reshape(-1, 24, 5, 5))
        t = F.relu(self.dc_bn3(t))
        t = self.dc_conv2(t)
        t = F.relu(self.dc_bn4(t))
        t = self.dc_conv3(t)
        t = torch.sigmoid(self.dc_bn5(t))

        return t

    def forward(self, t):
        mu, ln_var = self.encode(t)
        z = self.reparameter(mu, ln_var)
        x = self.decode(z)

        return x, mu, ln_var
