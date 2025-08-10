import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from ddpm.config import (
    DEVICE,
    img_size,
    hidden_dims,
    timestep_embedding_dim,
    n_timesteps,
    beta_minmax,
)
from ddpm.data import get_dataloaders
from ddpm.model import Denoiser
from ddpm.diffusion import Diffusion

def visualize_forward_process(n_images=1, n_steps=10):
    """
    Visualizes the forward diffusion process on a few images from the dataset.
    """
    # --- Setup ---
    train_loader, _ = get_dataloaders()
    
    model = Denoiser(
        image_resolution=img_size,
        hidden_dims=hidden_dims,
        diffusion_time_embedding_dim=timestep_embedding_dim,
        n_times=n_timesteps,
    ).to(DEVICE)

    diffusion = Diffusion(
        model,
        image_resolution=img_size,
        n_times=n_timesteps,
        beta_minmax=beta_minmax,
        device=DEVICE,
    ).to(DEVICE)

    # --- Get a batch of images ---
    images, _ = next(iter(train_loader))
    images = images[:n_images].to(DEVICE)

    # --- Create a list of timesteps to visualize ---
    t_steps = torch.linspace(0, n_timesteps - 1, n_steps, dtype=torch.long)

    # --- Apply forward diffusion at different timesteps ---
    noisy_images = []
    # Add original images first
    scaled_images = diffusion.scale_to_minus_one_to_one(images)
    
    for t in t_steps:
        t_tensor = torch.tensor([t] * n_images, device=DEVICE).long()
        noisy_img, _ = diffusion.make_noisy(scaled_images, t_tensor)
        noisy_images.append(diffusion.reverse_scale_to_zero_to_one(noisy_img.cpu()))

    # --- Plotting ---
    fig, axes = plt.subplots(n_images, n_steps + 1, figsize=(n_steps * 2, n_images * 2))
    if n_images == 1:
        axes = [axes]

    for i in range(n_images):
        # Original Image
        axes[i][0].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[i][0].set_title('Original')
        axes[i][0].axis('off')
        
        # Noisy Images
        for j in range(n_steps):
            axes[i][j + 1].imshow(noisy_images[j][i].squeeze(), cmap='gray')
            axes[i][j + 1].set_title(f't={t_steps[j]}')
            axes[i][j + 1].axis('off')

    plt.tight_layout()
    plt.savefig("forward_process_visualization.png")
    print("Forward process visualization saved to forward_process_visualization.png")

if __name__ == "__main__":
    visualize_forward_process(n_images=1, n_steps=10)