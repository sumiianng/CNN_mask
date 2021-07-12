import numpy as np
import pandas as pd

import torch
from torchvision import transforms

from modules.anime_dataset import AnimeDataset
from modules.train import train
from modules.line_plot import LinePlot
from modules.image_plot import ImagePlot


# dataset
train_set = AnimeDataset(
    root='.',
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
)

print(f"Gpu available : {torch.cuda.is_available()}")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for KL_coefficient in [1, 100, 0]:
    
    # train
    vae_network, manager = train(train_set, KL_coefficient=KL_coefficient, device=device)
    torch.save(vae_network.state_dict(), f'vae_network_{KL_coefficient}.ckpt')
    
    # learning curve
    df_all = pd.DataFrame(manager.run_data)
    ys = df_all['loss']
    LinePlot.learning_curve(df_all['epoch'], ys, title=f"Loss_{KL_coefficient}")
    
    # regenerated images
    vae_network.eval()
    
    sample_index = np.random.randint(len(train_set), size=64)
    original_images_tensor = train_set.data[sample_index]
    ImagePlot.sample_images(original_images_tensor, (8, 8), f'original_img_{KL_coefficient}')
    
    with torch.no_grad():
        regenerate_images_tensor, _, _ = vae_network(original_images_tensor.to(device))
    ImagePlot.sample_images(regenerate_images_tensor, (8, 8), f'regenerate_img_{KL_coefficient}')
    
    # generated images
    latent_dim = 20
    sample_num = 64
    zs = torch.randn(sample_num, latent_dim).to(device)
    with torch.no_grad():
        generate_images_tensor = vae_network.decode(zs)
        
    ImagePlot.sample_images(generate_images_tensor, (8, 8), f'generate_img_{KL_coefficient}')
    
    # change between two images
    sample_num = 8
    z_start = torch.randn(1, latent_dim)
    z_end = torch.randn(1, latent_dim)
    diff_of_z = (z_end-z_start) / (sample_num-1)
    z_tensor = torch.zeros(sample_num, latent_dim)
    for i in range(sample_num):
        z_tensor[i] = (z_start + diff_of_z*i).to(device)
    
    with torch.no_grad():
        images_change_tensor = vae_network.decode(z_tensor)
        
    ImagePlot.sample_images(images_change_tensor, (1, 8), f'images_change_{KL_coefficient}',
                              wspace=0.1) 
