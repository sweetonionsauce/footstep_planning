import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import heapq
import time
import matplotlib.pyplot as plt

# 全局常量定义
MAP_WIDTH = 3000
MAP_HEIGHT = 1000
MAX_STEP_LENGTH = 40  # 像素单位
MAX_TURN_ANGLE = 75  # 度
MAX_FOOT_ANGLE = 20  # 度
MIN_FOOT_DIST = 2  # 像素单位
MAX_FOOT_DIST = 10  # 像素单位
OBSERVATION_SIZE = 200  # 观测范围


# 复现MATLAB代码中的参数和逻辑
h0 = 30
h1 = h0 + h0 # 60
h2 = h1 + h0 # 90
h3 = h2 + h0 # 120
h4 = h3 - h0 # 90
h5 = h4 - h0 # 60
h6 = h5 - h0 # 30
h7 = h0 # 30

# 创建一个1000行3000列的数组，并全部初始化为h0
I = np.ones((MAP_HEIGHT, MAP_WIDTH)) * h0

# 按照MATLAB代码的逻辑划分区域并赋值
I[:, 1000:1060] = h1 # 注意Python索引是左闭右开，等效于MATLAB的1000:1060
I[:, 1060:1120] = h2
I[:, 1120:1300] = h3
I[:, 1300:1500] = h4
I[:, 1500:2000] = h5
I[:, 2000:3000] = h6 # 注意索引2000:3000在Python中是从2000到2999

# 添加随机噪声
noise = np.random.randn(MAP_HEIGHT, MAP_WIDTH)
I_noise = I + noise

# 为了模拟uint8转换，我们进行裁剪和类型转换（因为uint8范围是0-255）
# 先将数据缩放或裁剪到合适范围，避免溢出
# 这里简单处理：由于原始值 around 30-120, 噪声±1, 大概率在0-255内，直接转换
I_uint8 = np.clip(I_noise, 0, 255).astype(np.uint8)



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


'''
2.1 全局路径规划
使用A*算法在二维投影上计算从起点到终点的粗略路径。
地形高度作为代价因子（例如，高坡度区域代价高）。
A*输出一条路径点序列，提供全局方向指导局部规划。
'''


# 重写的A*算法，使用更高效的启发式和搜索策略
def A_star_planning(start, goal, height_map):
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


'''
2.2 局部落足点规划（在线）
'''


def partial_footstep_planning(current_bot_center_pos,current_bot_center_dir, current_lfoot_pos, current_rfoot_pos,
                              current_lfoot_normal, current_rfoot_normal, next_target_pos,
                              next_foot_to_move, height_map):
    """
    输入:
        current_bot_center_pos: 当前机器人中心位置 (x, y) - 原始坐标
        current_bot_center_dir: 当前机器人朝向向量 (dx, dy) - 单位向量
        current_lfoot_pos: 当前左足位置 (x, y) - 原始坐标
        current_rfoot_pos: 当前右足位置 (x, y) - 原始坐标
        current_lfoot_normal: 当前左足法向量 (nx, ny, nz) - 单位向量
        current_rfoot_normal: 当前右足法向量 (nx, ny, nz) - 单位向量
        next_target_pos: 下一个目标点坐标 (x, y) - 原始坐标
        next_foot_to_move: 下一步要移动的足 (0=左足, 1=右足)
        height_map: 地形高度图 (1000x3000数组)

    输出:
        new_foot_pos: 新足位置 (x, y) - 原始坐标
        new_foot_normal: 新足法向量 (nx, ny, nz) - 单位向量
        turning_angle: 转向角度 (度)
    """
    # 1. 提取观测区域
    obs_region = extract_observation_region(current_bot_center_pos, height_map)

    # 2. 生成候选落足点
    current_foot_pos = current_lfoot_pos if next_foot_to_move == 0 else current_rfoot_pos   #左脚0，右脚1
    other_foot_pos = current_rfoot_pos if next_foot_to_move == 0 else current_lfoot_pos
    candidate_positions = generate_candidate_positions(
        current_foot_pos,
        other_foot_pos,
        next_target_pos,
        current_bot_center_pos,
        current_bot_center_dir,
        height_map
    )

    # 3. 过滤满足约束的候选点
    valid_candidates = []
    for pos in candidate_positions:
        # 计算与另一足的距离
        other_foot_pos = current_rfoot_pos if next_foot_to_move == 0 else current_lfoot_pos
        dist = np.linalg.norm(np.array(pos) - np.array(other_foot_pos))

        # 检查距离约束
        if dist < MIN_FOOT_DIST or dist > MAX_FOOT_DIST:
            continue

        # 计算地形法向量
        foot_normal = compute_terrain_normal(height_map, pos)

        # 检查足平面夹角约束
        gravity_vector = np.array([0, 0, 1])
        dot_product = np.dot(foot_normal, gravity_vector)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        if angle_deg > MAX_FOOT_ANGLE:
            continue

        # 计算转向角
        turning_angle = compute_turning_angle(current_bot_center_pos, pos, next_target_pos)

        if abs(turning_angle) > MAX_TURN_ANGLE:
            continue

        # 所有约束满足，添加到有效候选
        valid_candidates.append({
            'position': pos,
            'normal': foot_normal,
            'turning_angle': turning_angle
        })

    # 4. 如果没有有效候选点，处理错误情况
    if not valid_candidates:
        # 处理无有效落足点的情况
        return None, None, None

    # 5. 使用强化学习选择最佳落足点
    best_candidate = reinforce_learning_selection(
        current_bot_center_pos,
        current_lfoot_pos,
        current_rfoot_pos,
        current_lfoot_normal,
        current_rfoot_normal,
        next_target_pos,
        next_foot_to_move,
        obs_region,
        valid_candidates
    )

    return best_candidate['position'], best_candidate['normal'], best_candidate['turning_angle']


def extract_observation_region(center_pos, height_map):
    """
    提取观测区域
    输入:
        center_pos: 中心位置 (x, y) - 原始坐标
        height_map: 完整高度图

    输出:
        obs_region: 观测区域高度图 (200x200数组) - 归一化到[0,1]
    """
    x, y = int(center_pos[0]), int(center_pos[1])
    x_start = max(0, x - OBSERVATION_SIZE // 2)
    x_end = min(MAP_WIDTH, x + OBSERVATION_SIZE // 2)
    y_start = max(0, y - OBSERVATION_SIZE // 2)
    y_end = min(MAP_HEIGHT, y + OBSERVATION_SIZE // 2)

    region = height_map[y_start:y_end, x_start:x_end]
    return normalize_height(region)


def generate_candidate_positions(current_foot_pos, other_foot_pos, next_target_pos,
                                 bot_center_pos, bot_direction, height_map):
    """
    生成候选落足点

    输入:
        current_foot_pos: 当前足位置 (x, y) - 原始坐标
        other_foot_pos: 另一足位置 (x, y) - 原始坐标
        next_target_pos: 下一个目标点 (x, y) - 原始坐标
        bot_center_pos: 机器人中心位置 (x, y) - 原始坐标
        bot_direction: 机器人朝向向量 (dx, dy) - 单位向量
        height_map: 地形高度图

    输出:
        candidate_positions: 候选位置列表 [(x1,y1), (x2,y2), ...] - 原始坐标
    """
    # 基本参数
    num_candidates = 100  # 生成的候选点数量
    min_step = 5  # 最小步长
    max_step = MAX_STEP_LENGTH  # 最大步长
    min_dist = MIN_FOOT_DIST  # 最小足间距
    max_dist = MAX_FOOT_DIST  # 最大足间距

    # 计算目标方向
    target_vector = np.array([next_target_pos[0] - bot_center_pos[0],
                              next_target_pos[1] - bot_center_pos[1]])
    target_dist = np.linalg.norm(target_vector)
    if target_dist > 1e-6:
        target_direction = target_vector / target_dist
    else:
        target_direction = np.array([1.0, 0.0])  # 默认方向

    # 计算机器人当前朝向与目标方向的夹角
    dot_product = np.dot(bot_direction, target_direction)
    angle_diff = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # 候选点列表
    candidate_positions = []

    # 生成候选点
    for i in range(num_candidates):
        # 1. 确定移动方向
        # 以目标方向为主，但加入随机扰动（最大±75°）
        max_angle = min(angle_diff + np.radians(75), np.pi)
        random_angle = np.random.uniform(-max_angle, max_angle)
        rot_mat = np.array([[np.cos(random_angle), -np.sin(random_angle)],
                            [np.sin(random_angle), np.cos(random_angle)]])
        move_direction = np.dot(rot_mat, target_direction)

        # 2. 确定移动距离 (在min_step和max_step之间)
        step_size = np.random.uniform(min_step, max_step)

        # 3. 计算候选点位置
        candidate_x = current_foot_pos[0] + move_direction[0] * step_size
        candidate_y = current_foot_pos[1] + move_direction[1] * step_size
        candidate_pos = (candidate_x, candidate_y)

        # 4. 检查是否在观测范围内 (200x200区域)
        if not (bot_center_pos[0] - 100 <= candidate_x <= bot_center_pos[0] + 100 and
                bot_center_pos[1] - 100 <= candidate_y <= bot_center_pos[1] + 100):
            continue

        # 5. 检查与另一足的距离
        dist_to_other = np.linalg.norm(np.array(candidate_pos) - np.array(other_foot_pos))
        if dist_to_other < min_dist or dist_to_other > max_dist:
            continue

        # 6. 检查地形高度差
        x1, y1 = int(current_foot_pos[0]), int(current_foot_pos[1])
        x2, y2 = int(candidate_x), int(candidate_y)

        # 确保位置在地图范围内
        if not (0 <= x1 < MAP_WIDTH and 0 <= y1 < MAP_HEIGHT and
                0 <= x2 < MAP_WIDTH and 0 <= y2 < MAP_HEIGHT):
            continue

        height1 = height_map[y1, x1]
        height2 = height_map[y2, x2]
        height_diff = abs(height2 - height1)

        # 如果高度差过大，跳过
        if height_diff > 20:  # 小于足平面夹角限制的20°
            continue

        # 添加到候选列表
        candidate_positions.append(candidate_pos)

    # 如果没有生成候选点，添加一个安全点
    if not candidate_positions:
        # 尝试在机器人当前位置附近添加一个点
        safe_point = (bot_center_pos[0] + 10, bot_center_pos[1])
        candidate_positions.append(safe_point)

    return candidate_positions


'''
2.3 强化学习选择
'''


def reinforce_learning_selection(current_bot_center_pos, current_lfoot_pos, current_rfoot_pos,
                                 current_lfoot_normal, current_rfoot_normal, next_target_pos,
                                 next_foot_to_move, obs_region, valid_candidates):
    """
    使用强化学习模型选择最佳落足点

    输入:
        current_bot_center_pos: 当前机器人中心位置 (x, y) - 原始坐标
        current_lfoot_pos: 当前左足位置 (x, y) - 原始坐标
        current_rfoot_pos: 当前右足位置 (x, y) - 原始坐标
        current_lfoot_normal: 当前左足法向量 (nx, ny, nz) - 单位向量
        current_rfoot_normal: 当前右足法向量 (nx, ny, nz) - 单位向量
        next_target_pos: 下一个目标点坐标 (x, y) - 原始坐标
        next_foot_to_move: 下一步要移动的足 (0=左足, 1=右足)
        obs_region: 观测区域高度图 (200x200数组) - 归一化到[0,1]
        valid_candidates: 有效候选点列表 [{
            'position': (x,y),
            'normal': (nx,ny,nz),
            'turning_angle': angle
        }]

    输出:
        best_candidate: 最佳候选点 (与输入valid_candidates中的元素相同)
    """
    # 1. 归一化所有位置信息
    norm_bot_center = normalize_position(current_bot_center_pos)
    norm_lfoot_pos = normalize_position(current_lfoot_pos)
    norm_rfoot_pos = normalize_position(current_rfoot_pos)
    norm_target_pos = normalize_position(next_target_pos)

    # 2. 准备神经网络输入
    # 注意：法向量已经是单位向量，不需要额外归一化
    state_vector = np.array([
        *norm_bot_center,
        *norm_lfoot_pos,
        *norm_rfoot_pos,
        *current_lfoot_normal,
        *current_rfoot_normal,
        *norm_target_pos,
        next_foot_to_move
    ])

    # 3. 对每个候选点准备输入
    candidate_q_values = []
    for candidate in valid_candidates:
        # 计算相对位移
        current_foot_pos = current_lfoot_pos if next_foot_to_move == 0 else current_rfoot_pos
        dx = candidate['position'][0] - current_foot_pos[0]
        dy = candidate['position'][1] - current_foot_pos[1]
        norm_disp = normalize_vector((dx, dy))

        # 组合完整输入
        input_vector = np.concatenate([
            state_vector,
            [*norm_disp],
            obs_region.flatten()  # 展平观测区域
        ])

        # 使用神经网络预测Q值
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
        q_value = footstep_net(input_tensor).item()
        candidate_q_values.append(q_value)

    # 4. 选择Q值最高的候选点
    best_idx = np.argmax(candidate_q_values)
    return valid_candidates[best_idx]


'''
2.3.1 计算奖励值
'''


def calculate_reward(current_state, next_state, action, is_terminal):
    """
    计算强化学习奖励

    输入:
        current_state: 当前状态字典
        next_state: 下一状态字典
        action: 采取的动作 (相对位移)
        is_terminal: 是否终止状态

    输出:
        reward: 奖励值
    """
    # 距离奖励 - 减少到目标的距离
    current_dist = np.linalg.norm(np.array(current_state['bot_center']) - np.array(current_state['target']))
    next_dist = np.linalg.norm(np.array(next_state['bot_center']) - np.array(next_state['target']))
    distance_reward = (current_dist - next_dist) * 10  # 缩放因子

    # 约束惩罚
    constraint_penalty = 0
    if next_state['foot_angle'] > MAX_FOOT_ANGLE:
        constraint_penalty -= 10
    if abs(next_state['turning_angle']) > MAX_TURN_ANGLE:
        constraint_penalty -= 10

    # 步数惩罚
    step_penalty = -0.1

    # 终止状态奖励
    terminal_reward = 100 if is_terminal and next_dist < 5 else 0

    # 总奖励
    reward = distance_reward + constraint_penalty + step_penalty + terminal_reward
    return reward


'''
2.3.2 NN架构
'''


class FootstepNet(nn.Module):
    def __init__(self):
        super(FootstepNet, self).__init__()
        # 状态向量部分 (位置、法向量等)
        self.fc_state = nn.Sequential(
            nn.Linear(15, 64),  # 15个状态特征
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 图像部分 (观测区域)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # 计算卷积层输出尺寸
        conv_output_size = 32 * (OBSERVATION_SIZE // 4) * (OBSERVATION_SIZE // 4)

        # 动作部分 (相对位移)
        self.fc_action = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        # 合并分支
        self.fc_combined = nn.Sequential(
            nn.Linear(64 + conv_output_size + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出Q值
        )

    def forward(self, x):
        # 输入结构: [state_vector (15), action_vector (2), obs_region_flattened (200 * 200)]
        state_vec = x[:, :15]
        action_vec = x[:, 15:17]
        obs_flat = x[:, 17:]

        # 处理状态向量
        state_out = self.fc_state(state_vec)

        # 处理观测区域 (恢复为2D)
        obs_2d = obs_flat.view(-1, 1, OBSERVATION_SIZE, OBSERVATION_SIZE)
        conv_out = self.conv(obs_2d)

        # 处理动作向量
        action_out = self.fc_action(action_vec)

        # 合并所有特征
        combined = torch.cat([state_out, conv_out, action_out], dim=1)
        q_value = self.fc_combined(combined)
        return q_value


# 实例化网络
footstep_net = FootstepNet()

'''
3. 地形法向量计算
'''


def compute_terrain_normal(height_map, foot_pos, foot_size=(5, 3)):
    """
    计算足覆盖区域的平均地形法向量

    输入:
        height_map: 地形高度图 (二维数组) - 原始尺度
        foot_pos: 足中心位置 (x, y) - 原始坐标
        foot_size: (length, width) 足尺寸，默认5x3

    输出:
        normal_vector: 法向量 (nx, ny, nz) - 单位向量
    """
    x, y = int(foot_pos[0]), int(foot_pos[1])
    half_len = foot_size[0] // 2
    half_wid = foot_size[1] // 2

    # 确保索引在范围内
    y_min = max(0, y - half_len)
    y_max = min(height_map.shape[0], y + half_len + 1)
    x_min = max(0, x - half_wid)
    x_max = min(height_map.shape[1], x + half_wid + 1)

    # 提取足区域高度
    foot_region = height_map[y_min:y_max, x_min:x_max]

    # 计算梯度
    gy, gx = np.gradient(foot_region)

    # 计算法向量
    nx = -np.mean(gx)
    ny = -np.mean(gy)
    nz = 1.0

    # 归一化
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    return (nx / norm, ny / norm, nz / norm)


'''
4. 机器人状态更新函数（新增）
这是核心修改：根据双足位置更新机器人位置和朝向
'''


def update_robot_state(new_foot_pos, next_foot_to_move,
                       current_lfoot_pos, current_rfoot_pos,
                       current_bot_center_dir):
    """
    根据双足位置更新机器人状态

    输入:
        new_foot_pos: 新移动的足位置 (x, y) - 原始坐标
        next_foot_to_move: 移动的足 (0=左足, 1=右足)
        current_lfoot_pos: 当前左足位置 (x, y) - 原始坐标
        current_rfoot_pos: 当前右足位置 (x, y) - 原始坐标
        current_bot_center_dir: 当前机器人朝向向量 (dx, dy) - 单位向量

    输出:
        new_bot_center_pos: 新机器人中心位置 (x, y) - 原始坐标
        new_bot_center_dir: 新机器人朝向向量 (dx, dy) - 单位向量
        new_lfoot_pos: 更新后的左足位置 (x, y) - 原始坐标
        new_rfoot_pos: 更新后的右足位置 (x, y) - 原始坐标
    """
    # 更新足位置
    if next_foot_to_move == 0:  # 移动左足
        new_lfoot_pos = new_foot_pos
        new_rfoot_pos = current_rfoot_pos
    else:  # 移动右足
        new_rfoot_pos = new_foot_pos
        new_lfoot_pos = current_lfoot_pos

    # 更新机器人中心位置（双足中点）
    new_bot_center_pos = (
        (new_lfoot_pos[0] + new_rfoot_pos[0]) / 2,
        (new_lfoot_pos[1] + new_rfoot_pos[1]) / 2
    )

    # 计算双足连线向量
    foot_vector = np.array([
        new_rfoot_pos[0] - new_lfoot_pos[0],
        new_rfoot_pos[1] - new_lfoot_pos[1]
    ])

    # 计算双足连线的垂直平分线方向（有两个可能方向）
    perpendicular1 = np.array([-foot_vector[1], foot_vector[0]])
    perpendicular2 = np.array([foot_vector[1], -foot_vector[0]])

    # 归一化
    perpendicular1 = perpendicular1 / np.linalg.norm(perpendicular1)
    perpendicular2 = perpendicular2 / np.linalg.norm(perpendicular2)

    # 选择与先前方向保持同侧的方向
    # 计算两个垂直向量与当前朝向的点积
    dot1 = np.dot(current_bot_center_dir, perpendicular1)
    dot2 = np.dot(current_bot_center_dir, perpendicular2)

    # 选择点积为正的方向（即与先前方向夹角小于90度）
    if dot1 > 0 and dot1 >= dot2:
        new_bot_center_dir = perpendicular1
    elif dot2 > 0 and dot2 > dot1:
        new_bot_center_dir = perpendicular2
    else:
        # 如果两个方向都与先前方向夹角大于90度
        # 选择夹角较小的那个（点积绝对值较大的）
        if abs(dot1) >= abs(dot2):
            new_bot_center_dir = perpendicular1
        else:
            new_bot_center_dir = perpendicular2

    # 归一化新方向
    new_bot_center_dir = new_bot_center_dir / np.linalg.norm(new_bot_center_dir)

    return new_bot_center_pos, new_bot_center_dir, new_lfoot_pos, new_rfoot_pos



'''
5. 转向角计算
'''
def compute_turning_angle(prev_direction, new_direction):
    """
    计算转向角（机器人朝向变化角度）

    输入:
        prev_direction: 先前朝向向量 (dx, dy) - 单位向量
        new_direction: 新朝向向量 (dx, dy) - 单位向量

    输出:
        turning_angle: 转向角度 (度)
    """
    # 计算点积（余弦值）
    dot_product = np.dot(prev_direction, new_direction)

    # 避免浮点误差导致超出[-1,1]范围
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # 计算角度（弧度）
    angle_rad = np.arccos(dot_product)

    # 转换为角度
    angle_deg = np.degrees(angle_rad)

    # 计算叉积确定方向（正值为逆时针，负值为顺时针）
    cross_product = prev_direction[0] * new_direction[1] - prev_direction[1] * new_direction[0]

    # 根据叉积符号确定转向方向
    if cross_product < 0:
        angle_deg = -angle_deg

    return angle_deg


# 辅助函数：从全局路径获取下一个目标点
def get_next_path_point(full_path, current_pos):
    """
    从全局路径中找到最接近当前位置的下一个目标点

    输入:
        full_path: 全局路径点列表 [(x1,y1), (x2,y2), ...]
        current_pos: 当前位置 (x, y)

    输出:
        next_target: 下一个目标点 (x, y)
    """
    # 找到最接近当前位置的路径点
    min_dist = float('inf')
    closest_idx = 0
    for i, point in enumerate(full_path):
        dist = np.linalg.norm(np.array(point) - np.array(current_pos))
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    # 选择下一个路径点（如果已经是最后一个，则保持不变）
    next_idx = min(closest_idx + 1, len(full_path) - 1)
    return full_path[next_idx]



# 主循环框架
def main_loop(start_pos, end_pos, height_map):
    # 初始化状态
    bot_center_pos = start_pos
    lfoot_pos = (start_pos[0] - 5, start_pos[1])  # 假设初始双足位置在起点两侧
    rfoot_pos = (start_pos[0] + 5, start_pos[1])
    lfoot_normal = compute_terrain_normal(height_map, lfoot_pos)
    rfoot_normal = compute_terrain_normal(height_map, rfoot_pos)
    next_foot_to_move = 0  # 0=左足, 1=右足

    # 初始机器人位置（双足中点）
    bot_center_pos = ((lfoot_pos[0] + rfoot_pos[0]) / 2,
                      (lfoot_pos[1] + rfoot_pos[1]) / 2)

    # 初始朝向（假设沿x轴正方向）
    bot_center_dir = np.array([1.0, 0.0])

    # 计算初始法向量
    lfoot_normal = compute_terrain_normal(height_map, lfoot_pos)
    rfoot_normal = compute_terrain_normal(height_map, rfoot_pos)

    next_foot_to_move = 0  # 0=左足, 1=右足

    # 全局路径规划
    next_target_pos, full_path = A_star_planning(bot_center_pos, end_pos, height_map)

    # 记录轨迹
    trajectory = []
    turning_angles = []  # 存储转向角

    while np.linalg.norm(np.array(bot_center_pos) - np.array(end_pos)) > 5:  # 距离终点小于5像素时停止
        # 局部落足点规划
        new_foot_pos, new_foot_normal, turning_angle = partial_footstep_planning(
            bot_center_pos, bot_center_dir,lfoot_pos, rfoot_pos,
            lfoot_normal, rfoot_normal, next_target_pos,
            next_foot_to_move, height_map
        )

        if new_foot_pos is None:
            print("无法找到有效落足点!")
            print("your coding skills suck")
            break

        '''
        # 更新足位置
        if next_foot_to_move == 0:
            lfoot_pos = new_foot_pos
            lfoot_normal = new_foot_normal
        else:
            rfoot_pos = new_foot_pos
            rfoot_normal = new_foot_normal

        # 更新机器人中心位置 (假设为双足中点)
        bot_center_pos = ((lfoot_pos[0] + rfoot_pos[0]) / 2,
                          (lfoot_pos[1] + rfoot_pos[1]) / 2)
        '''
        #记录之前的bot_center_dir,用于计算转向角
        prev_direction=bot_center_dir

        # 更新机器人状态
        (bot_center_pos, bot_center_dir,
         lfoot_pos, rfoot_pos) = update_robot_state(
            new_foot_pos,
            next_foot_to_move,
            lfoot_pos,
            rfoot_pos,
            bot_center_dir
        )

        # 更新法向量（只更新移动的足）
        if next_foot_to_move == 0:
            lfoot_normal = new_foot_normal
        else:
            rfoot_normal = new_foot_normal

        # 计算转向角（机器人朝向变化）
        turning_angle = compute_turning_angle(prev_direction, bot_center_dir)
        turning_angles.append(turning_angle)

        # 更新下一个要移动的足
        next_foot_to_move = 1 - next_foot_to_move

        # 记录轨迹点
        trajectory.append({
            'bot_center': bot_center_pos,
            'bot_direction': bot_center_dir,
            'lfoot_pos': lfoot_pos,
            'rfoot_pos': rfoot_pos,
            'turning_angle': turning_angle,
            #'foot_angle': np.degrees(np.arccos(np.dot(new_foot_normal, [0, 0, 1])))
        })

        # 更新下一个目标点 (如果接近当前目标点)
        '''
        if distance(bot_center_pos, next_target_pos) < MAX_STEP_LENGTH:
            next_target_pos = get_next_path_point(full_path, bot_center_pos)
        '''
        if np.linalg.norm(np.array(bot_center_pos) - np.array(next_target_pos)) < MAX_STEP_LENGTH:
            next_target_pos = get_next_path_point(full_path, bot_center_pos)
    # 输出结果
    return trajectory




'''
A* test
'''
# 测试优化后的A*算法
current_bot_center_pos = (100, 100)  # (x, y)
final_target_pos = (2800, 800)  # (x, y)

# 使用优化后的A*算法
next_target_pos, path = A_star_planning(
    current_bot_center_pos,
    final_target_pos,
    I_noise
)

print(f"下一个目标点: {next_target_pos}")
print(f"路径点数量: {len(path)}")
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

'''
落足点候选点测试
'''


def test_generate_candidate_positions():
    # 创建模拟地形
    height_map = np.zeros((MAP_HEIGHT, MAP_WIDTH))

    # 添加一些地形特征
    height_map[400:600, 1400:1600] = 50  # 一个高台
    height_map[200:400, 1800:2000] = -30  # 一个洼地

    # 初始化机器人状态
    bot_center_pos = (1500, 500)
    bot_direction = np.array([1.0, 0.0])  # 向东
    next_target_pos = (1600, 600)  # 东边1000像素处

    # 双足位置
    current_lfoot_pos = (1495, 500)
    current_rfoot_pos = (1505, 500)

    # 测试移动左足
    print("\n=== 测试移动左足 ===")
    candidates_left = generate_candidate_positions(
        current_foot_pos=current_lfoot_pos,
        other_foot_pos=current_rfoot_pos,
        next_target_pos=next_target_pos,
        bot_center_pos=bot_center_pos,
        bot_direction=bot_direction,
        height_map=height_map
    )

    print(f"生成候选点数量: {len(candidates_left)}")
    for i, pos in enumerate(candidates_left[:5]):  # 打印前5个点
        dist_to_other = np.linalg.norm(np.array(pos) - np.array(current_rfoot_pos))
        dist_to_target = np.linalg.norm(np.array(pos) - np.array(next_target_pos))
        print(
            f"候选点 {i + 1}: ({pos[0]:.1f}, {pos[1]:.1f}), 距离另一足: {dist_to_other:.1f}, 距离目标: {dist_to_target:.1f}")

    # 可视化左足候选点
    visualize_candidates(
        bot_center_pos=bot_center_pos,
        current_foot_pos=current_lfoot_pos,
        other_foot_pos=current_rfoot_pos,
        next_target_pos=next_target_pos,
        candidate_positions=candidates_left,
        height_map=height_map,
        title="move left foot candidate"
    )

    # 测试移动右足
    print("\n=== 测试移动右足 ===")
    candidates_right = generate_candidate_positions(
        current_foot_pos=current_rfoot_pos,
        other_foot_pos=current_lfoot_pos,
        next_target_pos=next_target_pos,
        bot_center_pos=bot_center_pos,
        bot_direction=bot_direction,
        height_map=height_map
    )

    print(f"生成候选点数量: {len(candidates_right)}")
    for i, pos in enumerate(candidates_right[:5]):  # 打印前5个点
        dist_to_other = np.linalg.norm(np.array(pos) - np.array(current_lfoot_pos))
        dist_to_target = np.linalg.norm(np.array(pos) - np.array(next_target_pos))
        print(
            f"候选点 {i + 1}: ({pos[0]:.1f}, {pos[1]:.1f}), 距离另一足: {dist_to_other:.1f}, 距离目标: {dist_to_target:.1f}")

    # 可视化右足候选点
    visualize_candidates(
        bot_center_pos=bot_center_pos,
        current_foot_pos=current_rfoot_pos,
        other_foot_pos=current_lfoot_pos,
        next_target_pos=next_target_pos,
        candidate_positions=candidates_right,
        height_map=height_map,
        title="move right foot candidate"
    )

    # 测试困难地形（机器人在高台边缘）
    print("\n=== 测试困难地形（高台边缘） ===")
    bot_center_pos = (1400, 500)
    current_lfoot_pos = (1395, 500)
    current_rfoot_pos = (1405, 500)
    next_target_pos = (1500, 500)  # 目标在高台上

    candidates_hard = generate_candidate_positions(
        current_foot_pos=current_lfoot_pos,
        other_foot_pos=current_rfoot_pos,
        next_target_pos=next_target_pos,
        bot_center_pos=bot_center_pos,
        bot_direction=bot_direction,
        height_map=height_map
    )

    print(f"生成候选点数量: {len(candidates_hard)}")
    if candidates_hard:
        for i, pos in enumerate(candidates_hard[:min(5, len(candidates_hard))]):
            dist_to_other = np.linalg.norm(np.array(pos) - np.array(current_rfoot_pos))
            print(f"候选点 {i + 1}: ({pos[0]:.1f}, {pos[1]:.1f}), 距离另一足: {dist_to_other:.1f}")

    # 可视化困难地形候选点
    visualize_candidates(
        bot_center_pos=bot_center_pos,
        current_foot_pos=current_lfoot_pos,
        other_foot_pos=current_rfoot_pos,
        next_target_pos=next_target_pos,
        candidate_positions=candidates_hard,
        height_map=height_map,
        title="high land edge candidate"
    )


def visualize_candidates(bot_center_pos, current_foot_pos, other_foot_pos,
                         next_target_pos, candidate_positions, height_map, title):
    plt.figure(figsize=(12, 6))

    # 显示高度图
    plt.imshow(height_map, cmap='terrain', extent=[0, MAP_WIDTH, 0, MAP_HEIGHT])
    plt.colorbar(label='height')

    # 标记关键位置
    plt.scatter(bot_center_pos[0], bot_center_pos[1], s=100, c='blue', marker='o', label='bot center')
    plt.scatter(current_foot_pos[0], current_foot_pos[1], s=80, c='red', marker='x', label='current foot')
    plt.scatter(other_foot_pos[0], other_foot_pos[1], s=80, c='green', marker='x', label='another foot')
    plt.scatter(next_target_pos[0], next_target_pos[1], s=100, c='purple', marker='*', label='target pos')

    # 标记候选点
    if candidate_positions:
        candidate_x = [p[0] for p in candidate_positions]
        candidate_y = [p[1] for p in candidate_positions]
        plt.scatter(candidate_x, candidate_y, s=50, c='cyan', alpha=0.7, label='candidate')

    # 添加观测范围框
    observation_box = plt.Rectangle(
        (bot_center_pos[0] - 100, bot_center_pos[1] - 100),
        200, 200,
        fill=False, color='yellow', linestyle='--', linewidth=1, label='observe area'
    )
    plt.gca().add_patch(observation_box)

    # 添加双足间距约束指示
    circle1 = plt.Circle(other_foot_pos, MIN_FOOT_DIST, fill=False, color='red', linestyle=':', linewidth=1,
                         label='min feet distance')
    circle2 = plt.Circle(other_foot_pos, MAX_FOOT_DIST, fill=False, color='green', linestyle=':', linewidth=1,
                         label='max feet distance')
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)

    # 设置标题和图例
    plt.title(title)
    plt.xlabel('X ')
    plt.ylabel('Y ')
    plt.legend(loc='upper right')

    # 设置视图范围（机器人中心周围200像素）
    plt.xlim(bot_center_pos[0] - 100, bot_center_pos[0] + 100)
    plt.ylim(bot_center_pos[1] - 100, bot_center_pos[1] + 100)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# 运行测试
test_generate_candidate_positions()

