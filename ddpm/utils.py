import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def show_image(x, idx):
    fig = plt.figure()
    plt.imshow(x[idx].transpose(0, 1).transpose(1, 2).detach().cpu().numpy())

def draw_sample_image(x, postfix):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)