import torch
from ddpm.config import (
    DEVICE,
    img_size,
    hidden_dims,
    timestep_embedding_dim,
    n_timesteps,
    beta_minmax,
    inference_batch_size,
)
from ddpm.model import Denoiser
from ddpm.diffusion import Diffusion
from ddpm.utils import draw_sample_image
import argparse

def sample(model_path):
    model = Denoiser(
        image_resolution=img_size,
        hidden_dims=hidden_dims,
        diffusion_time_embedding_dim=timestep_embedding_dim,
        n_times=n_timesteps,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    diffusion = Diffusion(
        model,
        image_resolution=img_size,
        n_times=n_timesteps,
        beta_minmax=beta_minmax,
        device=DEVICE,
    ).to(DEVICE)

    with torch.no_grad():
        generated_images = diffusion.sample(N=inference_batch_size)
    
    draw_sample_image(generated_images, "Generated Images")
    plt.savefig("generated_images.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the trained model checkpoint")
    args = parser.parse_args()
    sample(args.model_path)