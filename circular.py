
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


def plot_circular_structural_planes(csv_file, output_path=None, scale_factor=0.13, step=1,
                             view_elev=35, view_azim=45, default_radius=5.0,
                             position_spread=3.0, color_map='viridis', alpha=0.6):
    """
    生成圆形结构面3D可视化

    参数:
        csv_file (str): 必需的CSV文件路径
        output_path (str): 输出图片路径。如果为None，则基于输入文件名自动生成
        scale_factor (float): 半径缩放因子(默认0.3)
        step (int): 数据采样间隔(默认1)
        view_elev (float): 3D视图仰角(默认35)
        view_azim (float): 3D视图方位角(默认45)
        default_radius (float): 默认半径(当无trail length时使用，默认1.0)
        position_spread (float): 位置分布范围(默认3.0)
        color_map (str): 使用的颜色图谱(默认'viridis')
        alpha (float): 圆盘透明度(默认0.6)

    返回:
        str: 输出图片的保存路径

    异常:
        ValueError: 当CSV文件缺少必要列时
        FileNotFoundError: 当CSV文件不存在时
    """
    # 1. 数据加载与验证
    df = pd.read_csv(csv_file)

    # 检查必要列
    required_cols = ['dip direction', 'dip angle']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件中缺少必要列: {missing_cols}")

    # 2. 数据准备
    dip_directions = df['dip direction'].values
    dip_angles = df['dip angle'].values

    # 检查是否有trail length列
    has_trail_length = 'trail length' in df.columns
    if has_trail_length:
        trail_lengths = df['trail length'].values
        # 对trail length进行归一化处理，使其值在1-10范围内
        min_length = np.min(trail_lengths)
        max_length = np.max(trail_lengths)
        if max_length > min_length:  # 避免除以零
            trail_lengths = 1 + 9 * (trail_lengths - min_length) / (max_length - min_length)
    else:
        trail_lengths = np.ones_like(dip_directions) * default_radius

    # 3. 转换为弧度
    dip_directions_rad = np.radians(dip_directions)
    dip_angles_rad = np.radians(dip_angles)

    # 4. 创建图形
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 5. 颜色设置
    cmap = plt.get_cmap(color_map)
    norm = plt.Normalize(dip_angles.min(), dip_angles.max())

    # 6. 圆形生成函数
    def create_circle(radius, n_points=24):
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(x)
        return np.column_stack((x, y, z))

    # 7. 生成并绘制每个圆盘
    indices = range(0, len(dip_directions), step)
    for i in indices:
        radius = trail_lengths[i] * scale_factor
        circle = create_circle(radius)

        # 旋转计算
        rot_z = np.array([
            [np.cos(dip_directions_rad[i]), -np.sin(dip_directions_rad[i]), 0],
            [np.sin(dip_directions_rad[i]), np.cos(dip_directions_rad[i]), 0],
            [0, 0, 1]
        ])
        rot_y = np.array([
            [np.cos(dip_angles_rad[i]), 0, np.sin(dip_angles_rad[i])],
            [0, 1, 0],
            [-np.sin(dip_angles_rad[i]), 0, np.cos(dip_angles_rad[i])]
        ])

        # 应用旋转和平移
        circle = np.dot(circle, rot_z.T)
        circle = np.dot(circle, rot_y.T)
        circle += np.random.uniform(-position_spread, position_spread, size=3)

        poly = Poly3DCollection([circle], alpha=alpha)
        poly.set_facecolor(cmap(norm(dip_angles[i])))
        poly.set_edgecolor('k')
        poly.set_linewidth(0.3)
        ax.add_collection3d(poly)

    # 8. 设置图形属性
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    ax.set_xlabel('X (East)', fontsize=12)
    ax.set_ylabel('Y (North)', fontsize=12)
    ax.set_zlabel('Z (Up)', fontsize=12)

    # 标题设置
    file_name = os.path.basename(csv_file)
    base_name = os.path.splitext(file_name)[0]
    title = f'Circular Structural Planes - {base_name}\n(Showing {len(list(indices))}/{len(dip_directions)} planes)'
    if not has_trail_length:
        title += " (Using default radius)"
    ax.set_title(title, fontsize=14)

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Dip Angle (°)')
    cbar.ax.tick_params(labelsize=10)

    # 视角调整
    ax.view_init(elev=view_elev, azim=view_azim)

    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, f"{base_name}_circular_planes.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"圆形结构面图片已保存至: {output_path}")


if __name__ == "__main__":
    plot_circular_structural_planes(csv_file='Oernlia_set1.csv', output_path='./', scale_factor=0.13, step=1,
                                        view_elev=35, view_azim=45, default_radius=5.0,
                                        position_spread=3.0, color_map='viridis', alpha=0.6)