import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from base_method.boxplot.circular import plot_circular_structural_planes
from base_method.boxplot.GAN_boxplot_lib import GAN_plot_box_comparison1, GAN_plot_box_comparison2, GAN_plot_box_comparison3
from base_method.boxplot.GAN_histogram import plot_dip_direction_histogram, plot_dip_angle_histogram, plot_trail_length_histogram


def normalize_data(file_path, save_path):
    df = pd.read_csv(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(df)
    pd.DataFrame(normalized).to_csv(save_path, index=False)
    return torch.Tensor(normalized).float(), scaler


def create_dataloader(tensor_data, batch_size):
    return DataLoader(tensor_data, batch_size=batch_size, shuffle=True, drop_last=True)


def build_discriminator():
    return nn.Sequential(
        nn.Linear(3, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 1), nn.Sigmoid()
    )


def build_generator(z_dim):
    return nn.Sequential(
        nn.Linear(z_dim, 1024), nn.Tanh(),
        nn.Linear(1024, 512), nn.Tanh(),
        nn.Linear(512, 256), nn.Tanh(),
        nn.Linear(256, 128), nn.Tanh(),
        nn.Linear(128, 256), nn.Tanh(),
        nn.Linear(256, 256), nn.Tanh(),
        nn.Linear(256, 512), nn.Tanh(),
        nn.Linear(512, 3), nn.Sigmoid()
    )


def save_to_csv(directory, filename, data):
    file_path = os.path.join(directory, filename)
    data.to_csv(file_path, index=False)


def save_losses(directory, filename, losses):
    file_path = os.path.join(directory, filename)
    np.savetxt(file_path, losses, delimiter=",")


def save_model(directory, model, model_name):
    model_path = os.path.join(directory, model_name)
    torch.save(model.state_dict(), model_path)


def train_gan(input_path,
              output_path,
              z_dim=100,
              batch_size=8,
              num_epochs=200,
              plot=False):
    # 打印原始数据统计信息
    data = pd.read_csv(input_path)
    dip_direction = np.array(data[['dip direction']])
    dip_angle = np.array(data[['dip angle']])
    trail_length = np.array(data[['trail length']])
    print("原始数据统计信息:")
    print("倾向(dip direction): 均值 =", np.mean(dip_direction), "标准差 =", np.sqrt(np.var(dip_direction)))
    print("倾角(dip angle): 均值 =", np.mean(dip_angle), "标准差 =", np.sqrt(np.var(dip_angle)))
    print("迹长(trail length): 均值 =", np.mean(trail_length), "标准差 =", np.sqrt(np.var(trail_length)))

    os.makedirs(output_path, exist_ok=True)
    dataset, scaler = normalize_data(input_path, os.path.join(output_path, 'G_Standard_first_dataset.csv'))
    dataloader = create_dataloader(dataset, batch_size)
    D = build_discriminator()
    G = build_generator(z_dim)

    loss_func = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=6e-5)
    g_optimizer = optim.Adam(G.parameters(), lr=1e-6)

    g_losses, d_losses = [], []

    for epoch in range(num_epochs):
        print(f'--- Epoch {epoch + 1}/{num_epochs} ---')
        for real_data in dataloader:
            batch_size = real_data.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            real_out = D(real_data)
            d_loss_real = loss_func(real_out, real_labels)

            z = torch.randn(batch_size, z_dim)
            fake_data = G(z)
            fake_out = D(fake_data)
            d_loss_fake = loss_func(fake_out, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            d_losses.append(d_loss.item())

            for _ in range(5):
                z = torch.randn(batch_size, z_dim)
                fake_data = G(z)
                fake_out = D(fake_data)
                g_loss = loss_func(fake_out, real_labels)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                g_losses.append(g_loss.item())

    z = torch.randn(1000, z_dim)
    with torch.no_grad():
        fake_data = G(z)

    fake_data_np = fake_data.cpu().numpy()
    prediction = scaler.inverse_transform(fake_data_np)
    prediction = pd.DataFrame(prediction, columns=['dip direction', 'dip angle', 'trail length'])

    output_file = 'GAN_prediction.csv'
    save_to_csv(output_path, output_file, prediction)
    
    # 调用绘图函数
    csv_path = os.path.join(output_path, output_file)
    plot_circular_structural_planes(csv_path, output_path)
    GAN_plot_box_comparison1(input_path, csv_path, output_path)
    GAN_plot_box_comparison2(input_path, csv_path, output_path)
    GAN_plot_box_comparison3(input_path, csv_path, output_path)
    plot_dip_direction_histogram(input_path, csv_path, output_path)
    plot_dip_angle_histogram(input_path, csv_path, output_path)
    plot_trail_length_histogram(input_path, csv_path, output_path)

    # # 绘制训练损失曲线
    # plt.figure(figsize=(10, 5))
    # plt.plot(g_losses, label='Generator Loss')
    # plt.plot(d_losses, label='Discriminator Loss')
    # plt.legend()
    # plt.title('GAN Training Losses')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.savefig(os.path.join(output_path, 'GAN_training_loss.png'))
    # if plot:
    #     plt.show()
    



if __name__ == '__main__':
    train_gan(save_directory='./work')
