import torch
import torch.nn as nn
import math

#给扩散模型中的时间步（timestep）生成一个连续且平滑的高维向量表示
class SinusoidalPosEmb(nn.Module):
    """
    正弦位置嵌入 (Sinusoidal Positional Embedding)。
    用于将离散的时间步 t 转换为一个连续的、高维的向量表示。
    这种编码方式可以帮助模型理解时间步之间的相对关系。
    参考 "Attention Is All You Need" 论文中的位置编码方法。
    """
    def __init__(self, dim):
        """
        初始化嵌入层。
        Args:
            dim (int): 嵌入向量的维度。
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入的时间步张量，形状为 (N,)。
        Returns:
            torch.Tensor: 时间步的嵌入向量，形状为 (N, dim)。
        """
        device = x.device
        half_dim = self.dim // 2
        # 计算频率
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 计算位置编码
        emb = x[:, None] * emb[None, :]
        # 将 sin 和 cos 编码拼接起来
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConvBlock(nn.Conv2d):
    """
    一个自定义的卷积块，继承自 nn.Conv2d。
    它封装了卷积、可选的组归一化 (Group Normalization) 和激活函数。
    同时，它还支持残差连接 (residual connection) 和时间步嵌入的融合。
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                    stride=1, padding='same', dilation=1, groups=1, bias=True, gn=False, gn_groups=8):
        """
        初始化卷积块。
        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int): 卷积核大小。
            activation_fn (bool): 是否使用 SiLU 激活函数。
            drop_rate (float): Dropout 比率 (未使用)。
            stride (int): 步长。
            padding (str or int): 填充方式。'same' 表示保持输出尺寸不变。
            dilation (int): 扩张率。
            groups (int): 分组卷积的组数。
            bias (bool): 是否使用偏置。
            gn (bool): 是否使用组归一化。
            gn_groups (int): 组归一化的组数。
        """
        # 'same' padding 的计算
        if padding == 'same':
            padding = kernel_size // 2 * dilation

        super(ConvBlock, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)

        self.activation_fn = nn.SiLU() if activation_fn else None
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None
        
    def forward(self, x, time_embedding=None, residual=False):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入特征图 (N, C_in, H, W)。
            time_embedding (torch.Tensor, optional): 投影后的时间步嵌入 (N, C_out, 1, 1)。
            residual (bool): 是否使用残差连接。
        Returns:
            torch.Tensor: 输出特征图 (N, C_out, H, W)。
        """
        if residual:
            # 如果是残差块，先将时间嵌入加到输入上
            # 这是将时间信息融入模型的一种方式
            y = x + time_embedding
            # 进行卷积操作
            x_conv = super(ConvBlock, self).forward(y)
            # 添加残差连接
            y = y + x_conv
        else:
            # 非残差块，直接进行卷积
            y = super(ConvBlock, self).forward(x)
        
        # 应用组归一化和激活函数
        y = self.group_norm(y) if self.group_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        
        return y

class Denoiser(nn.Module):
    """
    Denoiser 模型是 DDPM 的核心，它是一个噪声预测器。
    给定一个加噪的图像 x_t 和时间步 t，该模型的目标是预测在时间步 t 添加到原始图像上的噪声。
    这个实现使用了一个带有残差连接和时间步嵌入的卷积网络，而不是一个完整的 U-Net 结构。
    """
    def __init__(self, image_resolution, hidden_dims=[256, 256], diffusion_time_embedding_dim = 256, n_times=1000):
        """
        初始化 Denoiser 模型。
        Args:
            image_resolution (tuple): 输入图像的分辨率 (H, W, C)。
            hidden_dims (list): 卷积层隐藏维度的列表。
            diffusion_time_embedding_dim (int): 扩散时间步嵌入的维度。
            n_times (int): 总的扩散时间步数。
        """
        super(Denoiser, self).__init__()
        
        _, _, img_C = image_resolution # 获取图像通道数
        
        # 1. 时间步嵌入层：将离散的时间步 t 转换为一个高维向量表示。
        self.time_embedding = SinusoidalPosEmb(diffusion_time_embedding_dim)
        
        # 2. 输入卷积层：将输入图像投影到第一个隐藏维度。
        self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=7)
        
        # 3. 时间步投影层：将时间步嵌入向量投影到与图像特征兼容的维度，以便后续融合。
        self.time_project = nn.Sequential(
                                 ConvBlock(diffusion_time_embedding_dim, hidden_dims[0], kernel_size=1, activation_fn=True),
                                 ConvBlock(hidden_dims[0], hidden_dims[0], kernel_size=1))
        
        # 4. 中间卷积层：一系列带有残差连接和扩张卷积的卷积块。
        self.convs = nn.ModuleList([ConvBlock(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=3)])
        
        for idx in range(1, len(hidden_dims)):
            # 使用扩张卷积来增大感受野，而不增加参数量
            self.convs.append(ConvBlock(hidden_dims[idx-1], hidden_dims[idx], kernel_size=3, dilation=3**((idx-1)//2),
                                                    activation_fn=True, gn=True, gn_groups=8))
                               
        # 5. 输出卷积层：将最终的特征图投影回原始图像的通道数，以预测噪声。
        self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_C, kernel_size=3)
        
    def forward(self, perturbed_x, diffusion_timestep):
        """
        前向传播过程。
        Args:
            perturbed_x (torch.Tensor): 加噪后的图像 (N, C, H, W)。
            diffusion_timestep (torch.Tensor): 扩散时间步 (N,)。
        Returns:
            torch.Tensor: 预测的噪声 (N, C, H, W)。
        """
        # (N, C, H, W)
        y = perturbed_x
        
        # 1. 计算时间步嵌入并进行投影
        # (N,) -> (N, dim)
        diffusion_embedding = self.time_embedding(diffusion_timestep)
        # (N, dim) -> (N, dim, 1, 1) -> (N, hidden_dim, 1, 1)
        diffusion_embedding = self.time_project(diffusion_embedding.unsqueeze(-1).unsqueeze(-2))
        
        # 2. 输入图像投影
        # (N, C, H, W) -> (N, hidden_dim, H, W)
        y = self.in_project(y)
        
        # 3. 通过一系列残差卷积块
        for i in range(len(self.convs)):
            # 在每个残差块中，将时间嵌入作为偏置项加到特征图上
            y = self.convs[i](y, diffusion_embedding, residual = True)
            
        # 4. 输出投影，得到预测的噪声
        # (N, hidden_dim, H, W) -> (N, C, H, W)
        y = self.out_project(y)
            
        return y