import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

class TorsionDataset(Dataset):
    """Dataset spécialisée pour des Torsions"""
    def __init__(self, torsion_data_path):
        """
        Args:
            torsion_data_path:扭转角数据文件路径 (.npy格式)
        """
        self.torsions = np.load(torsion_data_path)  # shape: (250000, 2)

        # 将角度转换为弧度（如果还不是的话）
        # mdtraj outputs angles in radians, so this check might be redundant
        # but keeping it won't hurt
        if np.max(np.abs(self.torsions)) > 2 * np.pi + 1e-6: # Add a small tolerance
            self.torsions = np.deg2rad(self.torsions)

        # 转换为torch tensor
        self.torsions = torch.FloatTensor(self.torsions)

    def __len__(self):
        return len(self.torsions)

    def __getitem__(self, idx):
        return self.torsions[idx]

def get_timestep_embedding(timesteps, embedding_dim):
    """
    标准的正弦位置编码用于时间步嵌入
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)

    # Ensure operations are done in float32
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class SimpleTorsionDiffusion(nn.Module):
    """改进的扭转角扩散模型"""
    def __init__(self, hidden_dim=128, num_layers=4, time_embed_dim=64):
        super().__init__()

        self.hidden_dim = hidden_dim # Store hidden_dim
        self.time_embed_dim = time_embed_dim
        self.num_layers = num_layers # Store num_layers

        # 时间嵌入层
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Input layer
        self.input_layer = nn.Linear(2, hidden_dim)
        self.input_activation = nn.SiLU()

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.SiLU())

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 2)


    def forward(self, x, t):
        """
        Args:
            x: 扭转角 [batch_size, 2]
            t: 时间步 [batch_size] (整数)
        """
        # Generate time embedding
        t_emb = get_timestep_embedding(t, self.time_embed_dim).to(x.device).to(x.dtype) # Cast time embedding to match input dtype
        t_emb = self.time_embed(t_emb) # Shape: [batch_size, hidden_dim]

        # Process input
        h = self.input_layer(x) # Shape: [batch_size, hidden_dim]
        h = self.input_activation(h) # Shape: [batch_size, hidden_dim]

        # Process hidden layers, adding time embedding before linear layers
        for i in range(0, len(self.hidden_layers), 2): # Iterate through linear/SiLU pairs
            linear_layer = self.hidden_layers[i]
            activation = self.hidden_layers[i+1]

            h = h + t_emb # Add time embedding before the linear layer
            h = linear_layer(h)
            h = activation(h)

        # Output layer
        h = self.output_layer(h) # Shape: [batch_size, 2]

        return h

class TorsionDDPM:
    def __init__(self, model, device='cuda', num_timesteps=1000, beta_schedule='linear'):
        self.model = model.to(device)
        self.device = device
        self.num_timesteps = num_timesteps

        # 设置噪声调度
        self.setup_noise_schedule(beta_schedule)

    def setup_noise_schedule(self, schedule_type='linear'):
        """设置噪声调度"""
        if schedule_type == 'linear':
            self.beta = torch.linspace(0.0001, 0.02, self.num_timesteps).to(self.device).to(torch.float32) # Ensure float32
        elif schedule_type == 'cosine':
            # 余弦调度
            s = 0.008
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps).to(self.device).to(torch.float32) # Ensure float32
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = torch.clip(betas, 0, 0.999).to(self.device).to(torch.float32) # Ensure float32

        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.ones(1).to(self.device).to(torch.float32), self.alpha_cumprod[:-1]]) # Ensure float32

        # 预计算一些有用的量
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha_cumprod = torch.sqrt(1.0 / self.alpha_cumprod)
        self.sqrt_recipm1_alpha_cumprod = torch.sqrt(1.0 / self.alpha_cumprod - 1)

    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：在时间步t添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)

        # Ensure indexing uses correct data type
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t]

        return (sqrt_alpha_cumprod_t.view(-1, 1) * x_start +
                sqrt_one_minus_alpha_cumprod_t.view(-1, 1) * noise), noise

    def predict_start_from_noise(self, x_t, t, noise_pred):
        """从噪声预测恢复原始数据"""
         # Ensure indexing uses correct data type
        return (self.sqrt_recip_alpha_cumprod[t].view(-1, 1) * x_t -
                self.sqrt_recipm1_alpha_cumprod[t].view(-1, 1) * noise_pred)

    def q_posterior(self, x_start, x_t, t):
        """计算后验分布的均值和方差"""
        # Ensure indexing uses correct data type
        beta_t = self.beta[t]
        alpha_cumprod_t = self.alpha_cumprod[t]
        alpha_cumprod_prev_t = self.alpha_cumprod_prev[t]
        alpha_t = self.alpha[t]

        posterior_variance = beta_t * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)
        posterior_log_variance_clipped = torch.log(torch.maximum(posterior_variance, torch.tensor(1e-20).to(self.device))) # Ensure tensor is on device

        posterior_mean_coef1 = beta_t * torch.sqrt(alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)
        posterior_mean_coef2 = (1.0 - alpha_cumprod_prev_t) * torch.sqrt(alpha_t) / (1.0 - alpha_cumprod_t)

        posterior_mean = (posterior_mean_coef1.view(-1, 1) * x_start +
                         posterior_mean_coef2.view(-1, 1) * x_t)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_t, t):
        """预测去噪后的均值和方差"""
        noise_pred = self.model(x_t, t)
        x_start_pred = self.predict_start_from_noise(x_t, t, noise_pred)

        # 对扭转角进行周期性约束
        x_start_pred = torch.remainder(x_start_pred + math.pi, 2 * math.pi) - math.pi

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start_pred, x_t, t)

        return model_mean, posterior_variance, posterior_log_variance, x_start_pred

    def p_sample(self, x_t, t):
        """单步去噪采样"""
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x_t, t)

        noise = torch.randn_like(x_t)
        # 当t=0时不添加噪声
        nonzero_mask = (t != 0).float().view(-1, 1)
        # Expand model_log_variance to match the shape of noise for element-wise multiplication
        model_std = torch.exp(0.5 * model_log_variance).view(-1, 1)
        return model_mean + nonzero_mask * model_std * noise

    def train_step(self, batch):
        """单步训练"""
        x_0 = batch.to(self.device)
        batch_size = x_0.shape[0]

        # 随机选择时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device, dtype=torch.long)

        # 添加噪声
        x_t, noise = self.q_sample(x_0, t)

        # 预测噪声
        noise_pred = self.model(x_t, t)

        # 计算损失
        loss = nn.MSELoss()(noise_pred, noise)
        return loss

    def train(self, dataloader, epochs=100, lr=1e-3):
        """训练循环"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()

                loss = self.train_step(batch)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            current_lr = scheduler.get_last_lr()[0]

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

    def sample(self, num_samples=1000, return_trajectory=False):
        """从训练好的模型采样新的扭转角"""
        self.model.eval()

        with torch.no_grad():
            # 从标准高斯分布开始
            x = torch.randn(num_samples, 2, device=self.device) # Default dtype is float32

            trajectory = [x.cpu().numpy()] if return_trajectory else None

            # 逆向扩散过程
            for t in reversed(range(self.num_timesteps)):
                t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                x = self.p_sample(x, t_tensor)

                if return_trajectory and t % 100 == 0:
                    trajectory.append(x.cpu().numpy())

            # 确保角度在[-π, π]范围内
            x = torch.remainder(x + math.pi, 2 * math.pi) - math.pi

        if return_trajectory:
            return x.cpu().numpy(), trajectory
        return x.cpu().numpy()

    def interpolate(self, x1, x2, num_steps=10):
        """在两个扭转角配置间插值"""
        self.model.eval()

        with torch.no_grad():
            # Ensure input tensors are float32
            x1 = torch.tensor(x1, device=self.device, dtype=torch.float32).unsqueeze(0)
            x2 = torch.tensor(x2, device=self.device, dtype=torch.float32).unsqueeze(0)

            # 添加噪声到中间时间步
            t = torch.full((1,), self.num_timesteps // 2, device=self.device, dtype=torch.long)
            x1_noisy, _ = self.q_sample(x1, t)
            x2_noisy, _ = self.q_sample(x2, t)

            # 在噪声空间中插值
            interpolated = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                # Ensure alpha is float32
                x_interp = (1.0 - float(alpha)) * x1_noisy + float(alpha) * x2_noisy

                # Go through reverse diffusion steps from the middle timestep
                for t_step in reversed(range(self.num_timesteps // 2)):
                    t_tensor = torch.full((1,), t_step, device=self.device, dtype=torch.long)
                    x_interp = self.p_sample(x_interp, t_tensor)

                interpolated.append(x_interp.cpu().numpy()[0])

        return np.array(interpolated)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    epochs = 200
    lr = 1e-3
    torsion_data_path = "alanine_torsions_from_1_xtc.npy"

    # 加载数据
    dataset = TorsionDataset(torsion_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print(f"Dataset size: {len(dataset)}")
    print(f"Data shape: {dataset.torsions.shape}")

    # 创建模型
    # Increase hidden_dim to 256 as specified in the original main function call
    model = SimpleTorsionDiffusion(hidden_dim=256, num_layers=6, time_embed_dim=128)
    ddpm = TorsionDDPM(model, device=device, num_timesteps=1000, beta_schedule='cosine')

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training on device: {device}")

    # 训练
    ddpm.train(dataloader, epochs=epochs, lr=lr)

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_timesteps': ddpm.num_timesteps,
        'beta_schedule': 'cosine'
    }, 'torsion_ddpm_model.pth')
    print("Model saved!")

    # 生成样本测试
    print("Generating samples...")
    samples = ddpm.sample(num_samples=100)
    samples_deg = np.rad2deg(samples)

    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample statistics (degrees):")
    print(f"Phi - Mean: {samples_deg[:, 0].mean():.2f}, Std: {samples_deg[:, 0].std():.2f}")
    print(f"Psi - Mean: {samples_deg[:, 1].mean():.2f}, Std: {samples_deg[:, 1].std():.2f}")

    # 保存结果
    np.save('generated_torsion_samples_ddpm.npy', samples)
    print("Generated samples saved!")

    # 测试插值功能
    print("Testing interpolation...")
    x1 = np.array([-2.0, -1.0])  # 一个构象
    x2 = np.array([1.0, 2.0])    # 另一个构象
    interpolated = ddpm.interpolate(x1, x2, num_steps=10)
    print(f"Interpolation shape: {interpolated.shape}")

    np.save('interpolated_torsions.npy', interpolated)
    print("Interpolation results saved!")

if __name__ == "__main__":
    main()