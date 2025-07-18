import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def Copula_plot_box_comparison1(real_data, simulation_data, output_path):
    """
    绘制观测数据与蒙特卡洛模拟数据的箱线图比较
    
    参数:
        observed_data (array-like): 观测数据数组
        monte_carlo_data (array-like): 蒙特卡洛模拟数据数组
        output_path (str): 图片保存路径，默认为"boxplot_comparison.png"
    
    返回:
        matplotlib.figure.Figure: 生成的图表对象
    """
    # 读取数据
    observed_data = pd.read_csv(real_data)['dip direction']
    Copula_data = pd.read_csv(simulation_data)['dip direction']

    # 创建DataFrame
    data = pd.DataFrame({
        'Value': pd.concat([pd.Series(observed_data), pd.Series(Copula_data)]),
        'Type': ['Observed'] * len(observed_data) + ['Generated by Copula'] * len(Copula_data),
        'Category': ['Observed'] * len(observed_data) + ['Copula'] * len(Copula_data)
    })

    # 设置图形风格
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(8, 6))

    # 定义颜色方案
    box_colors = ["#a6d854", "#8da0cb"]  # 浅绿色，浅蓝色
    point_colors = ["#1b9e77", "#377eb8"]  # 深绿色，深蓝色

    # 1. 绘制箱线图
    boxplot = sns.boxplot(
        x='Category',
        y='Value',
        hue='Type',
        data=data,
        palette=box_colors,
        width=0.6,
        showfliers=False,
        dodge=False
    )

    # 2. 绘制散点图
    for i, category in enumerate(['Observed', 'Copula']):
        subset = data[data['Category'] == category]
        sns.stripplot(
            x='Category',
            y='Value',
            data=subset,
            color=point_colors[i],
            alpha=0.7,
            size=4,
            jitter=0.2,
            ax=plt.gca()
        )

    # 修改x轴标签
    plt.xticks([0, 1], ['Observed', 'Copula'])

    # 调整y轴范围
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax * 1.1)

    # 添加标题和标签
    plt.xlabel("")
    plt.ylabel("Range (m)")

    # 调整图例位置
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in box_colors]
    plt.legend(
        handles,
        ['Observed', 'Generated by Copula'],
        loc='upper left',
        bbox_to_anchor=(0, 1),
        framealpha=1
    )

    # 隐藏多余的图例标题
    plt.gca().get_legend().set_title(None)

    # 指定要保存的文件名
    filename = "dip direction.png"
    # 组合完整路径
    full_output_path = os.path.join(output_path, filename)

    # 保存图形
    plt.tight_layout()
    fig.savefig(full_output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    return fig

def Copula_plot_box_comparison2(real_data, simulation_data, output_path):
    """
    绘制观测数据与蒙特卡洛模拟数据的箱线图比较
    
    参数:
        observed_data (array-like): 观测数据数组
        monte_carlo_data (array-like): 蒙特卡洛模拟数据数组
        output_path (str): 图片保存路径，默认为"boxplot_comparison.png"
    
    返回:
        matplotlib.figure.Figure: 生成的图表对象
    """
    # 读取数据
    observed_data = pd.read_csv(real_data)['dip angle']
    Copula_data = pd.read_csv(simulation_data)['dip angle']

    # 创建DataFrame
    data = pd.DataFrame({
        'Value': pd.concat([pd.Series(observed_data), pd.Series(Copula_data)]),
        'Type': ['Observed'] * len(observed_data) + ['Generated by Copula'] * len(Copula_data),
        'Category': ['Observed'] * len(observed_data) + ['Copula'] * len(Copula_data)
    })

    # 设置图形风格
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(8, 6))

    # 定义颜色方案
    box_colors = ["#a6d854", "#8da0cb"]  # 浅绿色，浅蓝色
    point_colors = ["#1b9e77", "#377eb8"]  # 深绿色，深蓝色

    # 1. 绘制箱线图
    boxplot = sns.boxplot(
        x='Category',
        y='Value',
        hue='Type',
        data=data,
        palette=box_colors,
        width=0.6,
        showfliers=False,
        dodge=False
    )

    # 2. 绘制散点图
    for i, category in enumerate(['Observed', 'Copula']):
        subset = data[data['Category'] == category]
        sns.stripplot(
            x='Category',
            y='Value',
            data=subset,
            color=point_colors[i],
            alpha=0.7,
            size=4,
            jitter=0.2,
            ax=plt.gca()
        )

    # 修改x轴标签
    plt.xticks([0, 1], ['Observed', 'Copula'])

    # 调整y轴范围
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax * 1.1)

    # 添加标题和标签
    plt.xlabel("")
    plt.ylabel("Range (m)")

    # 调整图例位置
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in box_colors]
    plt.legend(
        handles,
        ['Observed', 'Generated by Copula'],
        loc='upper left',
        bbox_to_anchor=(0, 1),
        framealpha=1
    )

    # 隐藏多余的图例标题
    plt.gca().get_legend().set_title(None)

    # 指定要保存的文件名
    filename = "dip angle.png"
    # 组合完整路径
    full_output_path = os.path.join(output_path, filename)

    # 保存图形
    plt.tight_layout()
    fig.savefig(full_output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    return fig

def Copula_plot_box_comparison3(real_data, simulation_data, output_path):
    """
    绘制观测数据与蒙特卡洛模拟数据的箱线图比较
    
    参数:
        observed_data (array-like): 观测数据数组
        monte_carlo_data (array-like): 蒙特卡洛模拟数据数组
        output_path (str): 图片保存路径，默认为"boxplot_comparison.png"
    
    返回:
        matplotlib.figure.Figure: 生成的图表对象
    """
    # 读取数据
    observed_data = pd.read_csv(real_data)['trail length']
    Copula_data = pd.read_csv(simulation_data)['trail length']

    # 创建DataFrame
    data = pd.DataFrame({
        'Value': pd.concat([pd.Series(observed_data), pd.Series(Copula_data)]),
        'Type': ['Observed'] * len(observed_data) + ['Generated by Copula'] * len(Copula_data),
        'Category': ['Observed'] * len(observed_data) + ['Copula'] * len(Copula_data)
    })

    # 设置图形风格
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(8, 6))

    # 定义颜色方案
    box_colors = ["#a6d854", "#8da0cb"]  # 浅绿色，浅蓝色
    point_colors = ["#1b9e77", "#377eb8"]  # 深绿色，深蓝色

    # 1. 绘制箱线图
    boxplot = sns.boxplot(
        x='Category',
        y='Value',
        hue='Type',
        data=data,
        palette=box_colors,
        width=0.6,
        showfliers=False,
        dodge=False
    )

    # 2. 绘制散点图
    for i, category in enumerate(['Observed', 'Copula']):
        subset = data[data['Category'] == category]
        sns.stripplot(
            x='Category',
            y='Value',
            data=subset,
            color=point_colors[i],
            alpha=0.7,
            size=4,
            jitter=0.2,
            ax=plt.gca()
        )

    # 修改x轴标签
    plt.xticks([0, 1], ['Observed', 'Copula'])

    # 调整y轴范围
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax * 1.1)

    # 添加标题和标签
    plt.xlabel("")
    plt.ylabel("Range (m)")

    # 调整图例位置
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in box_colors]
    plt.legend(
        handles,
        ['Observed', 'Generated by Copula'],
        loc='upper left',
        bbox_to_anchor=(0, 1),
        framealpha=1
    )

    # 隐藏多余的图例标题
    plt.gca().get_legend().set_title(None)

    # 指定要保存的文件名
    filename = "trail length.png"
    # 组合完整路径
    full_output_path = os.path.join(output_path, filename)

    # 保存图形
    plt.tight_layout()
    fig.savefig(full_output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    return fig