import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base_method.boxplot.circular import plot_circular_structural_planes
from base_method.boxplot.MC_boxplot_lib import MC_plot_box_comparison1, MC_plot_box_comparison2, MC_plot_box_comparison3
from base_method.boxplot.MC_histogram import plot_dip_direction_histogram, plot_dip_angle_histogram, plot_trail_length_histogram

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.unicode_minus'] = False


def simulate_monte_carlo_data(input_path, output_path, plot=False):
    data = pd.read_csv(input_path)
    dip_direction = np.array(data[['dip direction']])
    dip_angle = np.array(data[['dip angle']])
    trail_length = np.array(data[['trail length']])
    print("原始数据统计信息:")
    print("倾向(dip direction): 均值 =", np.mean(dip_direction), "标准差 =", np.sqrt(np.var(dip_direction)))
    print("倾角(dip angle): 均值 =", np.mean(dip_angle), "标准差 =", np.sqrt(np.var(dip_angle)))
    print("迹长(trail length): 均值 =", np.mean(trail_length), "标准差 =", np.sqrt(np.var(trail_length)))
    mean = np.mean(trail_length)
    variance = np.var(trail_length)
    mu = np.log((mean ** 2) / np.sqrt(variance + mean ** 2))
    sigma = np.sqrt(np.log(variance / mean ** 2 + 1))
    num_samples = len(data)
    simulated_direction = np.random.normal(np.mean(dip_direction), np.sqrt(np.var(dip_direction)), num_samples)
    simulated_angle = np.random.normal(np.mean(dip_angle), np.sqrt(np.var(dip_angle)), num_samples)
    simulated_trail_length = np.random.lognormal(mu, sigma, num_samples)

    # # 创建四个单独的图形
    # # 图1: 倾向 vs 倾角
    # plt.figure(figsize=(8, 6))
    # plt.scatter(dip_direction, dip_angle, alpha=0.5, label='原始数据')
    # plt.scatter(simulated_direction, simulated_angle, alpha=0.5, label='模拟数据')
    # plt.title('倾向 vs 倾角')
    # plt.xlabel('倾向 (dip direction)')
    # plt.ylabel('倾角 (dip angle)')
    # plt.legend()
    # plt.savefig(os.path.join(output_path, 'MC_dip_direction_vs_angle.png'))
    # if plot:
    #     plt.show()
    # plt.close()
    
    # # 图2: 倾向 vs 迹长
    # plt.figure(figsize=(8, 6))
    # plt.scatter(dip_direction, trail_length, alpha=0.5, label='原始数据')
    # plt.scatter(simulated_direction, simulated_trail_length, alpha=0.5, label='模拟数据')
    # plt.title('倾向 vs 迹长')
    # plt.xlabel('倾向 (dip direction)')
    # plt.ylabel('迹长 (trail length)')
    # plt.legend()
    # plt.savefig(os.path.join(output_path, 'MC_dip_direction_vs_length.png'))
    # if plot:
    #     plt.show()
    # plt.close()
    
    # # 图3: 倾角 vs 迹长
    # plt.figure(figsize=(8, 6))
    # plt.scatter(dip_angle, trail_length, alpha=0.5, label='原始数据')
    # plt.scatter(simulated_angle, simulated_trail_length, alpha=0.5, label='模拟数据')
    # plt.title('倾角 vs 迹长')
    # plt.xlabel('倾角 (dip angle)')
    # plt.ylabel('迹长 (trail length)')
    # plt.legend()
    # plt.savefig(os.path.join(output_path, 'MC_dip_angle_vs_length.png'))
    # if plot:
    #     plt.show()
    # plt.close()
    
    # # 图4: 迹长分布对比
    # plt.figure(figsize=(8, 6))
    # plt.hist(trail_length, bins=30, alpha=0.5, label='原始数据')
    # plt.hist(simulated_trail_length, bins=30, alpha=0.5, label='模拟数据')
    # plt.title('迹长分布对比')
    # plt.xlabel('迹长 (trail length)')
    # plt.ylabel('频数')
    # plt.legend()
    # plt.savefig(os.path.join(output_path, 'MC_trail_length_distribution.png'))
    # if plot:
    #     plt.show()
    # plt.close()
    simulated_df = pd.DataFrame({
        'dip direction': simulated_direction,
        'dip angle': simulated_angle,
        'trail length': simulated_trail_length
    })
    csv_path = os.path.join(output_path, 'Montecarlo_prediction.csv')
    simulated_df.to_csv(csv_path, index=False)
    plot_circular_structural_planes(csv_path,output_path)
    MC_plot_box_comparison1(input_path, csv_path, output_path)
    MC_plot_box_comparison2(input_path, csv_path, output_path)
    MC_plot_box_comparison3(input_path, csv_path, output_path)
    plot_dip_direction_histogram(input_path, csv_path, output_path)
    plot_dip_angle_histogram(input_path, csv_path, output_path)
    plot_trail_length_histogram(input_path, csv_path, output_path)


if __name__ == "__main__":
    simulate_monte_carlo_data('Oernlia_set1.csv',
                              r'C:\Users\zgml\Documents\WeChat Files\wxid_ze82r7d0fucy22\FileStorage\File\2025-04\软件包')
