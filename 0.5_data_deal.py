import yaml
import os
import csv

# 读取29_servo_config.yaml配置文件
with open('29_servo_config.yaml', 'r', encoding='utf-8') as f:
    servo_config = yaml.safe_load(f)

# 获取电机的start_deg和end_deg
servo_ranges = {}
for i in range(29):
    key = f'A{i}'
    start_deg = servo_config[key]['start_deg']
    end_deg = servo_config[key]['end_deg']
    servo_ranges[i] = (start_deg, end_deg)

# 表情文件路径
expression_dir = 'expression29_anchor'
expression_files = [
    'angry.yaml',
    'disgust.yaml',
    'fear.yaml',
    'happy.yaml',
    'sad.yaml',
    'surprise.yaml',
    'neutral.yaml'
]

# 存储所有结果
all_ratios = []

# 处理每个表情文件
for filename in expression_files:
    filepath = os.path.join(expression_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        expression_data = yaml.safe_load(f)
    
    # 计算每个电机参数的占比
    ratios = []
    for i in range(29):
        start_deg, end_deg = servo_ranges[i]
        value = expression_data[i]
        
        # 计算占比
        if end_deg != start_deg:
            ratio = (value - start_deg) / (end_deg - start_deg)
            # 确保比例值在0到1之间
            ratio = max(0, min(1, ratio))
        else:
            ratio = 0
        # 保留两位小数
        ratios.append(round(ratio, 2))
    
    # 存储结果，去掉文件后缀
    expression_name = filename.replace('.yaml', '')
    all_ratios.append([expression_name] + ratios)

# 将结果写入CSV文件
with open('expression_anchor.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['expression'] + [f'servo_{i}' for i in range(29)])  # 写入表头
    writer.writerows(all_ratios)

# 打印结果
for row in all_ratios:
    print(row)