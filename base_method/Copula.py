import os.path
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from copulae import GaussianCopula
import matplotlib.pyplot as plt

from base_method.boxplot.circular import plot_circular_structural_planes
from base_method.boxplot.Copula_boxplot_lib import Copula_plot_box_comparison1, Copula_plot_box_comparison2, Copula_plot_box_comparison3
from base_method.boxplot.Copula_histogram import plot_dip_direction_histogram, plot_dip_angle_histogram, plot_trail_length_histogram

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.unicode_minus'] = False


def simulate_copula_data(input_path, output_path, plot=False):
    # 打印原始数据统计信息
    data = pd.read_csv(input_path)
    x = data['dip direction'].values
    y = data['dip angle'].values
    z = data['trail length'].values
    print("原始数据统计信息:")
    print("倾向(dip direction): 均值 =", np.mean(x), "标准差 =", np.sqrt(np.var(x)))
    print("倾角(dip angle): 均值 =", np.mean(y), "标准差 =", np.sqrt(np.var(y)))
    print("迹长(trail length): 均值 =", np.mean(z), "标准差 =", np.sqrt(np.var(z)))

    u = rankdata(x, method='average') / (len(x) + 1)
    v = rankdata(y, method='average') / (len(y) + 1)
    w = rankdata(z, method='average') / (len(z) + 1)
    copula = GaussianCopula(dim=3)
    copula.fit(np.column_stack([u, v, w]))
    num_samples = len(x)
    simulated_uvw = copula.random(num_samples)
    direction = np.quantile(x, simulated_uvw[:, 0])
    angle = np.quantile(y, simulated_uvw[:, 1])
    length = np.quantile(z, simulated_uvw[:, 2])

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # ax1.scatter(x, y, alpha=0.6)
    # ax1.set_title('原始数据')
    # ax1.set_xlabel('dip direction')
    # ax1.set_ylabel('dip angle')

    # ax2.scatter(direction, angle, alpha=0.6)
    # ax2.set_title('模拟数据')
    # ax2.set_xlabel('dip direction')
    # ax2.set_ylabel('dip angle')
    # if plot:
    #     plt.show()
    # path = os.path.join(output_path, 'C_simulated_data.png')
    # plt.savefig(path)

    simulated_df = pd.DataFrame({
        'dip direction': direction,
        'dip angle': angle,
        'trail length': length
    })
    path = os.path.join(output_path, 'Copula_prediction.csv')
    simulated_df.to_csv(path, index=False)
    
    # 调用绘图函数
    plot_circular_structural_planes(path, output_path)
    Copula_plot_box_comparison1(input_path, path, output_path)
    Copula_plot_box_comparison2(input_path, path, output_path)
    Copula_plot_box_comparison3(input_path, path, output_path)
    plot_dip_direction_histogram(input_path, path, output_path)
    plot_dip_angle_histogram(input_path, path, output_path)
    plot_trail_length_histogram(input_path, path, output_path)

if __name__ == "__main__":
    simulate_copula_data('Oernlia_set1.csv', 'work')
