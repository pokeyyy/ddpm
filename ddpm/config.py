import torch

# Model Hyperparameters
dataset_path = './dataset'
cuda = True
DEVICE = torch.device("cuda:0" if cuda else "cpu")
dataset = 'MNIST'
img_size = (28, 28, 1)  # (width, height, channels)
#时间步长的嵌入维度
timestep_embedding_dim = 256
n_layers = 8
hidden_dim = 256
#时间步数
n_timesteps = 1000
#噪声的调节参数范围
beta_minmax = [1e-4, 2e-2]
train_batch_size = 128
inference_batch_size = 64
lr = 5e-5
epochs = 200
seed = 1234
hidden_dims = [hidden_dim for _ in range(n_layers)]

# Seed
torch.manual_seed(seed)