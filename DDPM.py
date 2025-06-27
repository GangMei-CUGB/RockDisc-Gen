import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from circular import plot_circular_structural_planes

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True


class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=1024):
        super(MLPDiffusion, self).__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(3, num_units))
        for _ in range(7):
            self.linears.append(nn.ReLU())
            self.linears.append(nn.Linear(num_units, num_units))
        self.linears.append(nn.ReLU())
        self.linears.append(nn.Linear(num_units, 3))
        self.step_embeddings = nn.ModuleList([nn.Embedding(n_steps, num_units) for _ in range(3)])

    def forward(self, x, t):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
        return x


def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]
    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0).unsqueeze(-1)
    a = alphas_bar_sqrt[t]
    aml = one_minus_alphas_bar_sqrt[t]
    e = torch.randn_like(x_0)
    x = x_0 * a + e * aml
    output = model(x, t.squeeze(-1))
    return ((e - output) ** 2).mean()


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    t = torch.tensor([t])
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - coeff * eps_theta)
    z = torch.randn_like(x)
    return mean + betas[t].sqrt() * z


def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, data_num):
    cur_x = torch.randn((data_num, 3))
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
    return cur_x


def run_diffusion_training(input_csv, output_dir, data_num=900, num_epochs=200, plot=True):
    start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(df)
    pd.DataFrame(normalized).to_csv(os.path.join(output_dir, 'D_Standard_dataset.csv'), index=False)

    data = pd.DataFrame(normalized)
    joints_data = data.values
    dataset = torch.tensor(joints_data, dtype=torch.float32)

    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1.]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    model = MLPDiffusion(num_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    losses = []

    print("开始训练...")
    for epoch in range(num_epochs):
        # 严格的epoch计数验证
        if epoch >= num_epochs:
            break
            
        # 添加进度条
        progress = (epoch + 1) / num_epochs * 100
        print(f"\rTraining progress: {progress:.1f}% ({epoch+1}/{num_epochs})", end="")
        
        for batch_x in dataloader:
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        # 每epoch都打印，添加更清晰的标识
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Loss = {loss.item():.6f}")

    print("训练完成！")

    x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt, data_num)
    prediction = scaler.inverse_transform(x_seq.detach().numpy())
    prediction = prediction[prediction.min(axis=1) >= 0]  # 去除任何负值

    prediction_df = pd.DataFrame(prediction, columns=['dip direction', 'dip angle', 'trail length'])
    prediction_df.to_csv(os.path.join(output_dir, 'prediction.csv'), index=False)
    plot_circular_structural_planes(os.path.join(output_dir, 'prediction.csv'), output_dir)
    np.savetxt(os.path.join(output_dir, 'loss.csv'), losses, delimiter=',')

    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'DDPM_training_loss.png'))

    if plot:
        plt.show()

    end = time.time()
    print(f"训练与采样总耗时：{end - start:.2f} 秒")


if __name__ == '__main__':
    run_diffusion_training(
        input_csv='Oernlia_set1.csv',
        output_dir='work',
        data_num=900,
        num_epochs=200,
        plot=True
    )
