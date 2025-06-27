import os.path
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from copulae import GaussianCopula
import matplotlib.pyplot as plt
from circular import plot_circular_structural_planes

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.unicode_minus'] = False


def simulate_copula_data(input_path, output_path, plot=False):
    data = pd.read_csv(input_path)
    x = data['dip direction'].values
    y = data['dip angle'].values
    u = rankdata(x, method='average') / (len(x) + 1)
    v = rankdata(y, method='average') / (len(y) + 1)
    copula = GaussianCopula(dim=2)
    copula.fit(np.column_stack([u, v]))
    num_samples = len(x)
    simulated_uv = copula.random(num_samples)
    direction = np.quantile(x, simulated_uv[:, 0])
    angle = np.quantile(y, simulated_uv[:, 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(x, y, alpha=0.6)
    ax1.set_title('原始数据')
    ax1.set_xlabel('dip direction')
    ax1.set_ylabel('dip angle')

    ax2.scatter(direction, angle, alpha=0.6)
    ax2.set_title('模拟数据')
    ax2.set_xlabel('dip direction')
    ax2.set_ylabel('dip angle')
    if plot:
        plt.show()
    path = os.path.join(output_path, 'C_simulated_data.png')
    plt.savefig(path)

    simulated_df = pd.DataFrame({'dip direction': direction, 'dip angle': angle})
    path = os.path.join(output_path, 'C_simulated_data.csv')
    simulated_df.to_csv(path, index=False)
    plot_circular_structural_planes(path, output_path)


if __name__ == "__main__":
    simulate_copula_data('Oernlia_set1.csv',
                         r'C:\Users\zgml\Documents\WeChat Files\wxid_ze82r7d0fucy22\FileStorage\File\2025-04\软件包')
