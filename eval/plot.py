import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 原始数据字符串
data_str = """MATH	1	2	4	8	16		1	2	4	8	16		AIME2024	1	2	4	8	16		1	2	4	8	16			AIME2025	1	2	4	8	16		1	2	4	8	16		AMC2023	1	2	4	8	16		1	2	4	8	16		GPQA	1	2	4	8	16		1	2	4	8	16
grpo	77.12%	82.34%	86.12%	88.90%	91.00%		76.88%	77.16%	77.21%	77.23%	77.12%		grpo	27.08%	35.11%	42.01%	48.80%	50.00%		33.33%	31.67%	28.33%	30.42%	27.08%			grpo	9.38%	13.47%	18.07%	23.40%	30.00%		6.67%	5.00%	6.67%	7.08%	9.38%		grpo	61.09%	69.52%	75.77%	80.64%	85.00%		62.50%	66.25%	65.00%	61.56%	61.09%		grpo	20.47%	30.80%	41.75%	51.63%	60.27%		20.31%	19.31%	20.15%	20.42%	20.47%
grpo +82	76.73%	81.20%	84.56%	87.04%	88.96%		76.78%	76.88%	76.73%	76.80%	76.73%		grpo +82	26.04%	33.61%	40.08%	45.85%	50.00%		23.33%	23.33%	26.67%	25.42%	26.04%			grpo +82	11.25%	15.83%	20.74%	25.50%	30.00%		6.67%	6.67%	11.67%	11.67%	11.25%		grpo +82	63.59%	68.00%	71.22%	74.17%	77.50%		65.00%	65.00%	64.38%	63.44%	63.59%		grpo +82	20.01%	29.53%	39.33%	47.97%	54.69%		22.10%	21.21%	21.99%	20.62%	20.01%
grpo +first20	77.13%	82.44%	86.21%	88.98%	91.16%		77.34%	77.41%	77.26%	77.24%	77.13%		grpo +first20	24.79%	32.97%	40.22%	45.50%	50.00%		26.67%	21.67%	22.50%	23.75%	24.79%			grpo +first20	10.42%	14.61%	19.47%	24.56%	26.67%		10.00%	10.00%	11.67%	10.83%	10.42%		grpo +first20	59.84%	67.77%	73.76%	78.04%	82.50%		60.00%	55.00%	58.13%	59.38%	59.84%		grpo +first20	18.93%	28.28%	38.45%	47.87%	56.03%		19.42%	19.20%	18.36%	18.55%	18.93%
grpo+adv+entropy	76.92%	81.45%	84.78%	87.35%	89.50%		77.14%	76.93%	76.90%	76.86%	76.92%		grpo+adv+entropy	27.29%	34.31%	39.52%	44.17%	50.00%		33.33%	35.00%	30.00%	26.67%	27.29%			grpo+adv+entropy	10.00%	14.19%	18.98%	23.51%	26.67%		3.33%	6.67%	7.50%	8.33%	10.00%		grpo+adv+entropy	61.41%	67.00%	70.26%	73.16%	77.50%		60.00%	60.00%	60.00%	60.63%	61.41%		grpo+adv+entropy	19.59%	29.00%	38.88%	47.91%	55.80%		18.75%	17.86%	19.08%	19.25%	19.59%
cpo_neg0.1_allq_dpqactor_noclip_hasallgood	76.03%	83.01%	87.60%	90.78%	93.04%		75.76%	75.89%	75.99%	76.11%	76.03%		cpo_neg0.1_allq_dpqactor_noclip_hasallgood	24.38%	34.83%	43.84%	49.71%	53.33%		26.67%	25.00%	21.67%	21.67%	21.67%			cpo_neg0.1_allq_dpqactor_noclip_hasallgood	10.42%	15.69%	21.27%	27.08%	33.33%		3.33%	8.33%	12.50%	12.50%	13.13%		cpo_neg0.1_allq_dpqactor_noclip_hasallgood	59.06%	68.90%	75.92%	82.31%	87.50%		57.50%	60.00%	64.38%	61.25%	59.38%		cpo_neg0.1_allq_dpqactor_noclip_hasallgood	18.46%	28.68%	40.37%	51.80%	62.05%		18.53%	17.41%	17.30%	18.33%	18.42%
cpo_neg0.1pos0.1_allq_dpqactor_noclip_hasallgood	75.22%	82.74%	87.49%	90.69%	92.90%		75.50%	75.03%	75.08%	75.11%	75.22%		cpo_neg0.1pos0.1_allq_dpqactor_noclip_hasallgood	24.58%	35.44%	44.49%	50.91%	56.67%		26.67%	31.67%	26.67%	25.42%	24.58%			cpo_neg0.1pos0.1_allq_dpqactor_noclip_hasallgood	8.96%	14.72%	21.81%	29.40%	36.67%		6.67%	10.00%	9.17%	8.33%	8.96%		cpo_neg0.1pos0.1_allq_dpqactor_noclip_hasallgood	58.75%	68.85%	76.64%	82.53%	87.50%		52.50%	55.00%	58.13%	56.25%	58.75%		cpo_neg0.1pos0.1_allq_dpqactor_noclip_hasallgood	18.00%	28.10%	39.93%	52.05%	63.17%		20.09%	19.98%	18.19%	17.77%	18.00%
cpo_pos0.1_allq_dpqactor_noclip_hasallgood	76.50%	83.21%	87.70%	90.82%	93.04%		76.12%	76.34%	76.54%	76.61%	76.50%		cpo_pos0.1_allq_dpqactor_noclip_hasallgood	22.50%	33.81%	43.85%	50.60%	56.67%		20.00%	20.00%	21.67%	21.25%	22.50%			cpo_pos0.1_allq_dpqactor_noclip_hasallgood	9.17%	14.03%	19.69%	26.31%	33.33%		13.33%	13.33%	11.67%	10.42%	9.17%		cpo_pos0.1_allq_dpqactor_noclip_hasallgood	59.69%	69.69%	78.71%	86.61%	95.00%		55.00%	57.50%	60.00%	58.44%	59.69%		cpo_pos0.1_allq_dpqactor_noclip_hasallgood	18.30%	28.50%	40.13%	51.18%	59.82%		16.52%	17.52%	17.13%	17.75%	18.30%"""

def parse_data(data_str):
    """
    解析数据字符串，返回结构化数据
    返回格式: {task_name: {model_name: [values for k=1,2,4,8,16]}}
    """
    lines = data_str.strip().split('\n')
    header_line = lines[0]
    data_lines = lines[1:]
    
    # 解析header，找到所有任务名称和它们的位置
    header_parts = header_line.split('\t')
    tasks = []
    current_task = None
    task_start_idx = {}
    
    for idx, part in enumerate(header_parts):
        if part and not part.isdigit():
            current_task = part
            tasks.append(current_task)
            task_start_idx[current_task] = idx
    
    # 为每个任务构建数据结构
    result = defaultdict(lambda: defaultdict(list))
    
    for line in data_lines:
        parts = line.split('\t')
        model_name = None
        current_task = None
        
        for idx, part in enumerate(parts):
            # 识别模型名称
            for task in tasks:
                if idx >= task_start_idx[task]:
                    current_task = task
            if idx == 0:
                model_name = part.strip()
                # 确定当前属于哪个任务
            # 提取百分比数值
            elif task_start_idx[current_task] < idx <= task_start_idx[current_task] + 5:
                if part.endswith('%') and model_name and current_task:
                    value = float(part.rstrip('%'))
                else:
                    value = 0
                result[current_task][model_name].append(value)
    
    return result

def plot_task(task_name, models_data, k_values=[1, 2, 4, 8, 16]):
    """
    为单个任务绘制折线图（使用对数刻度横坐标）
    """
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for idx, (model_name, values) in enumerate(models_data.items()):
        if len(values) == len(k_values):
            plt.plot(k_values, values, 
                    marker=markers[idx % len(markers)], 
                    color=colors[idx % len(colors)],
                    linewidth=2, 
                    markersize=8,
                    label=model_name,
                    alpha=0.8)
    
    # 使用对数刻度
    plt.xscale('log', base=2)
    
    plt.xlabel('pass@k', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title(f'{task_name} Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(k_values, k_values)  # 显示实际的k值
    plt.tight_layout()
    
    return plt

def plot_all_tasks(data_str):
    """
    解析数据并为每个任务绘制折线图
    """
    data = parse_data(data_str)
    k_values = [1, 2, 4, 8, 16]
    
    # 为每个任务创建单独的图
    # for task_name, models_data in data.items():
    #     plot_task(task_name, models_data, k_values)
        # plt.savefig(f'{task_name}_performance.png', dpi=300, bbox_inches='tight')
        # plt.show()
    
    # 创建一个包含所有子图的大图
    n_tasks = len(data)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f']
    
    for task_idx, (task_name, models_data) in enumerate(data.items()):
        ax = axes[task_idx]
        for model_idx, (model_name, values) in enumerate(models_data.items()):
            if len(values) == len(k_values):
                ax.plot(k_values, values,
                       marker=markers[model_idx % len(markers)],
                       color=colors[model_idx % len(colors)],
                       linewidth=2,
                       markersize=6,
                       label=model_name,
                       alpha=0.8)
        
        # 使用对数刻度
        ax.set_xscale('log', base=2)
        ax.set_xticks(k_values)
        ax.set_xticklabels(k_values)
        
        ax.set_xlabel('pass@k', fontsize=10, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'{task_name}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # 隐藏多余的子图
    for idx in range(n_tasks, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('all_tasks_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# 运行绘图
plot_all_tasks(data_str)