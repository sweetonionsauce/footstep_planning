import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import heapq
import time
import matplotlib.pyplot as plt

# 全局常量定义
MAP_WIDTH = 3000  # X轴方向宽度
MAP_HEIGHT = 1000  # Y轴方向高度
MAX_STEP_LENGTH = 40  # 像素单位
MAX_TURN_ANGLE = 75  # 度
MAX_FOOT_ANGLE = 20  # 度
MIN_FOOT_DIST = 2  # 像素单位
MAX_FOOT_DIST = 10  # 像素单位
OBSERVATION_SIZE = 200  # 观测范围

MAX_TOLERANCE = 40  # 允许的A*最后位置和目标的最大误差距离

# 复现MATLAB代码中的参数和逻辑
h0 = 30
h1 = h0 + h0  # 60
h2 = h1 + h0  # 90
h3 = h2 + h0  # 120
h4 = h3 - h0  # 90
h5 = h4 - h0  # 60
h6 = h5 - h0  # 30
h7 = h0  # 30

# 创建一个1000行3000列的数组（Y轴1000，X轴3000）
# 注意：数组索引为 [y, x]
I = np.ones((MAP_HEIGHT, MAP_WIDTH)) * h0

# 按照MATLAB代码的逻辑划分区域并赋值
I[:, 1000:1060] = h1  # X轴方向1000-1060
I[:, 1060:1120] = h2  # X轴方向1060-1120
I[:, 1120:1300] = h3  # X轴方向1120-1300
I[:, 1300:1500] = h4  # X轴方向1300-1500
I[:, 1500:2000] = h5  # X轴方向1500-2000
I[:, 2000:3000] = h6  # X轴方向2000-3000


# 添加随机噪声
noise = np.random.randn(MAP_HEIGHT, MAP_WIDTH)
I_noise = I + noise

# 为了模拟uint8转换，我们进行裁剪和类型转换
I_uint8 = np.clip(I_noise, 0, 255).astype(np.uint8)

# 全局设置
PATH_STEP_SIZE = 80  # 路径点之间的最小距离（像素）
HEIGHT_DIFF_THRESHOLD = 30  # 允许的最大高度差（设为不可通行）


# 归一化工具函数
def normalize_position(pos, map_width=MAP_WIDTH, map_height=MAP_HEIGHT):
    """归一化位置坐标到[-1, 1]范围"""
    x_norm = (pos[0] / map_width) * 2 - 1
    y_norm = (pos[1] / map_height) * 2 - 1
    return (x_norm, y_norm)


def denormalize_position(norm_pos, map_width=MAP_WIDTH, map_height=MAP_HEIGHT):
    """反归一化位置坐标"""
    x = (norm_pos[0] + 1) / 2 * map_width
    y = (norm_pos[1] + 1) / 2 * map_height
    return (x, y)


def normalize_vector(vec, max_value=MAX_STEP_LENGTH):
    """归一化向量到[-1, 1]范围"""
    return (vec[0] / max_value, vec[1] / max_value)


def denormalize_vector(norm_vec, max_value=MAX_STEP_LENGTH):
    """反归一化向量"""
    return (norm_vec[0] * max_value, norm_vec[1] * max_value)


def normalize_height(height_map):
    """归一化高度图到[0, 1]范围"""
    min_val = np.min(height_map)
    max_val = np.max(height_map)
    return (height_map - min_val) / (max_val - min_val)

# 重写的A*算法，使用更高效的启发式和搜索策略
def optimized_A_star(start, goal, height_map):
    """
    优化的A*路径规划算法，使用更有效的启发式和搜索策略
    """
    # 地图尺寸 (height_map.shape = [y_size, x_size])
    y_size, x_size = height_map.shape
    print(f"起点: {start}, 终点: {goal}")
    print(f"地图尺寸: X={x_size}, Y={y_size}")

    # 定义节点类
    class Node:
        __slots__ = ('position', 'parent', 'g', 'h', 'f')

        def __init__(self, position, parent=None):
            self.position = position  # (x, y)
            self.parent = parent
            self.g = 0  # 从起点到当前节点的实际代价
            self.h = 0  # 启发式代价
            self.f = 0  # 总代价

        def __lt__(self, other):
            return self.f < other.f

    # 方向设置 - 8个方向
    directions = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]

    # 对角线移动的距离
    diag_cost = math.sqrt(2)

    # 改进的启发式函数 - 结合曼哈顿距离和方向引导
    def heuristic(pos):
        # 基本曼哈顿距离
        dx = abs(goal[0] - pos[0])
        dy = abs(goal[1] - pos[1])

        # 方向引导因子 - 鼓励向目标方向移动
        dir_factor = 1.0
        if goal[0] > pos[0]:  # 目标在右边
            dir_factor += 0.1 * (goal[0] - pos[0]) / x_size
        elif goal[0] < pos[0]:  # 目标在左边
            dir_factor += 0.1 * (pos[0] - goal[0]) / x_size

        if goal[1] > pos[1]:  # 目标在下边
            dir_factor += 0.1 * (goal[1] - pos[1]) / y_size
        elif goal[1] < pos[1]:  # 目标在上边
            dir_factor += 0.1 * (pos[1] - goal[1]) / y_size

        return (dx + dy) * dir_factor

    # 移动代价函数
    def move_cost(current_pos, next_pos):
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # 边界检查
        if not (0 <= next_pos[0] < x_size and 0 <= next_pos[1] < y_size):
            return float('inf')

        # 获取高度
        x1, y1 = int(current_pos[0]), int(current_pos[1])
        x2, y2 = int(next_pos[0]), int(next_pos[1])
        height1 = height_map[y1, x1]
        height2 = height_map[y2, x2]

        # 高度差限制
        height_diff = abs(height2 - height1)
        if height_diff > 30:
            return float('inf')

        # 对角线移动额外成本
        cost = distance
        if dx != 0 and dy != 0:  # 对角线移动
            cost = diag_cost * max(abs(dx), abs(dy))

        return cost + height_diff * 0.1

    # 初始化
    start_node = Node(start)
    start_node.h = heuristic(start)
    start_node.f = start_node.h

    open_list = []
    heapq.heappush(open_list, start_node)

    closed_set = set()
    closed_set.add(start)

    # 节点扩展计数器
    nodes_expanded = 0
    start_time = time.time()

    # 目标接近阈值
    goal_threshold = 5

    while open_list:
        current_node = heapq.heappop(open_list)
        nodes_expanded += 1

        # 进度报告
        if nodes_expanded % 10000 == 0:
            dist_to_goal = math.sqrt(
                (goal[0] - current_node.position[0]) ** 2 +
                (goal[1] - current_node.position[1]) ** 2
            )
            elapsed = time.time() - start_time
            print(f"节点: {nodes_expanded}, 位置: {current_node.position}, "
                  f"距离目标: {dist_to_goal:.1f}, 耗时: {elapsed:.2f}s")

        # 检查是否到达目标
        if (abs(current_node.position[0] - goal[0]) <= goal_threshold and
                abs(current_node.position[1] - goal[1]) <= goal_threshold):
            path = []
            node = current_node
            while node:
                path.append(node.position)
                node = node.parent
            path.reverse()
            print(f"找到路径! 路径长度: {len(path)}, 总节点数: {nodes_expanded}")
            return path[1] if len(path) > 1 else path[0], path

        # 探索邻居节点
        for dx, dy in directions:
            new_x = current_node.position[0] + dx
            new_y = current_node.position[1] + dy
            new_pos = (new_x, new_y)

            # 跳过已访问节点
            if new_pos in closed_set:
                continue

            # 计算移动成本
            cost = move_cost(current_node.position, new_pos)
            if cost == float('inf'):
                continue

            # 创建新节点
            new_node = Node(new_pos, current_node)
            new_node.g = current_node.g + cost
            new_node.h = heuristic(new_pos)
            new_node.f = new_node.g + new_node.h

            # 添加到开放列表
            heapq.heappush(open_list, new_node)
            closed_set.add(new_pos)

    # 未找到路径
    print(f"未找到路径! 总节点数: {nodes_expanded}")
    return start, [start]


# 测试优化后的A*算法
current_bot_center_pos = (100, 100)  # (x, y)
final_target_pos = (2800, 800)  # (x, y)

# 使用优化后的A*算法
next_target_pos, path = optimized_A_star(
    current_bot_center_pos,
    final_target_pos,
    I_noise
)

print(f"下一个目标点: {next_target_pos}")
print(f"路径点数量: {len(path)}")
if len(path) > 5:
    print(f"前5个路径点: {path[:5]}")
else:
    print(f"所有路径点: {path}")
n=1
selected_points=[]
while n<len(path) :
    if n%10==0:
        selected_points.append(path[n])
    n+=1
print(selected_points)

# 为了模拟uint8转换，我们进行裁剪和类型转换（因为uint8范围是0-255）
# 先将数据缩放或裁剪到合适范围，避免溢出
# 这里简单处理：由于原始值 around 30-120, 噪声±1, 大概率在0-255内，直接转换
I_uint8 = np.clip(I_noise, 0, 255).astype(np.uint8)

# 将路径点转换为数组格式
path_points = np.array(path)

# 创建可视化图像
plt.figure(figsize=(15, 5))  # 宽15英寸，高5英寸（匹配3000x1000的比例）

# 显示高度图（使用灰度）
plt.imshow(I_uint8, cmap='gray', origin='lower', extent=[0, MAP_WIDTH, 0, MAP_HEIGHT])

# 绘制路径点 - 使用红色标记
plt.scatter(path_points[:, 0], path_points[:, 1],
            s=1,  # 点的大小
            c='red',
            alpha=0.6,  # 半透明效果
            label='Path Points')

# 标记起点和终点
plt.scatter([current_bot_center_pos[0]], [current_bot_center_pos[1]],
            s=100, c='green', marker='o', label='Start')
plt.scatter([final_target_pos[0]], [final_target_pos[1]],
            s=100, c='blue', marker='x', label='End')

# 添加标题和图例
plt.title(f'Path Visualization (Length: {len(path)} points)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()

# 保存图像
plt.savefig('path_visualization.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()