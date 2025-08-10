import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm

from ddpm.config import (
    DEVICE,
    img_size,
    hidden_dims,
    timestep_embedding_dim,
    n_timesteps,
    beta_minmax,
    lr,
    epochs,
)
from ddpm.data import get_dataloaders
from ddpm.model import Denoiser
from ddpm.diffusion import Diffusion
from ddpm.utils import count_parameters

def train():
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

    optimizer = Adam(diffusion.parameters(), lr=lr)
    denoising_loss = nn.MSELoss()

    print("Number of model parameters: ", count_parameters(diffusion))

    print("Start training DDPMs...")
    model.train()

    for epoch in range(epochs):
        noise_prediction_loss = 0
        for batch_idx, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            x = x.to(DEVICE)

            noisy_input, epsilon, pred_epsilon = diffusion(x)
            loss = denoising_loss(pred_epsilon, epsilon)

            noise_prediction_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(
            "\tEpoch",
            epoch + 1,
            "complete!",
            "\tDenoising Loss: ",
            noise_prediction_loss / batch_idx,
        )
        torch.save(model.state_dict(), f"ddpm_model_epoch_{epoch}.pth")

    print("Finish!!")

if __name__ == "__main__":
    train()