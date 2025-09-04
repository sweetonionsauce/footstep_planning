import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

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
I = np.ones((1000, 3000)) * h0

# 按照MATLAB代码的逻辑划分区域并赋值
I[:, 1000:1060] = h1 # 注意Python索引是左闭右开，等效于MATLAB的1000:1060
I[:, 1060:1120] = h2
I[:, 1120:1300] = h3
I[:, 1300:1500] = h4
I[:, 1500:2000] = h5
I[:, 2000:3000] = h6 # 注意索引2000:3000在Python中是从2000到2999

# 添加随机噪声
noise = np.random.randn(1000, 3000)
I_noise = I + noise

# 为了模拟uint8转换，我们进行裁剪和类型转换（因为uint8范围是0-255）
# 先将数据缩放或裁剪到合适范围，避免溢出
# 这里简单处理：由于原始值 around 30-120, 噪声±1, 大概率在0-255内，直接转换
I_uint8 = np.clip(I_noise, 0, 255).astype(np.uint8)

'''
# 可视化
plt.figure(figsize=(15, 5)) # 设置画布大小，因为图像很宽

plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.colorbar()
plt.title('Generated Map (Without Noise)')

plt.subplot(1, 2, 2)
plt.imshow(I_uint8, cmap='gray')
plt.colorbar()
plt.title('Generated Map (With Noise, uint8)')

plt.tight_layout()
plt.show()
'''

'''
背景：双足机器人在不平坦的路面上运动。双足交替移动会受到以下因素制约：步长、足平面法线方向与重力方向夹角、观测范围、实际落足点与规划落足点的偏差。给定起点和终点位置，如何在线规划落足点的位置？
地图信息
# 创建一个1000行3000列的数组，并全部初始化为h0
I = np.ones((1000, 3000)) * h0

# 按照MATLAB代码的逻辑划分区域并赋值
I[:, 1000:1060] = h1 # 注意Python索引是左闭右开，等效于MATLAB的1000:1060
I[:, 1060:1120] = h2
I[:, 1120:1300] = h3
I[:, 1300:1500] = h4
I[:, 1500:2000] = h5
I[:, 2000:3000] = h6 # 注意索引2000:3000在Python中是从2000到2999

# 添加随机噪声
noise = np.random.randn(1000, 3000)
I_noise = I + noise
'''
'''
输入条件：（1）三维地形图；（2）足尺寸3×5；（3）步长最大值40；双足间距大于2，小于10；
（4）观测范围：运动方向前方200×200区域；（5）最大转向角小于+/-75°；
（5）足平面法线与重力方向夹角小于+/-20°；（6）起点和终点的投影坐标。
输出要求：（1）双足落足点序列（足中心位置坐标和足平面法线方向与重力方向夹角，可视化）；
（2）轨迹长度；（3）转向角曲线；（4）足平面法线方向与重力方向夹角曲线；
（5）实验数据分析和安全性分析
'''
'''
我的分析：
运动：
所以说这实际上是一个水平方向向量（起点代表机器人核心位置，方向代表它的方向），
加上两个法向量，代表两个足平面的法向量和七点，区域是3x5（暂且当作投影的区域也是3x5的），
在水平向量前200x200区域内，足选择前方（与该足原位置距离小于40，与另一足当前位置大于2小于10位置（先暂且不考虑足的体积））
然后先：假设更新足的位置，再基于假设更新水平向量，计算更新后的转向角和足平面法线与竖直方向夹角，如果满足小于75°小于20°
就真更新，否则假设不成立重新选点
'''
'''
记录双足落足点序列（每个记录为：（位置坐标（2维）+法向量（3维）+左/右编号（1位））），水平向量序列。
'''
'''
可以用神经网络更新的部分是什么？
是选择每个落足点的策略，需要输入的信息应该包括三个向量当前位置（两法向量，配两坐标（2维），加上一个水平向量（3维，其中高度先预留不用）和坐标（2维）），
目前做选择的足的编号（1位），当前最近的目标点的位置（2维）（我打算用A*算法寻路），前面200*200区域内的地图信息，
整合成一个tensor作为神经网络输入，损失函数应该考虑：错误决策点惩罚（假设不符合），步长越短惩罚越高（为了尽快），到达目标点的总步数越多惩罚越高，
N步内水平向量位置离目标位置的距离缩减越小惩罚越高，网络输出应该是下一步该足位置（法向量对应水平坐标）。
'''

#######正式开始痛苦
'''
2.1 全局路径规划
使用A*算法在二维投影上计算从起点到终点的粗略路径。地形高度作为代价因子（例如，高坡度区域代价高）。
A*输出一条路径点序列，提供全局方向指导局部规划。
'''
#A*寻路，输入当前位置（对应水平向量中心点），最终目标点位置，步长(步长应大于40像素)，输出下一个目标点的坐标
def A_star_planning(current_bot_center_pos,final_target_pos,one_step_length,map):
    pass
    #return next_target_pos

'''
2.2 局部落足点规划（在线）
'''
def Partial_footstep_planning(current_bot_center_pos,current_lfoot_pos,current_rfoot_pos,current_lfoot_dir,current_rfoot_dir,next_target_pos,next_foot_to_move):

    pass
    #return new_foot_pos

'''
2.3 强化学习设置
'''
def Reinforce_learning_config(partial_map,current_bot_center_pos,current_lfoot_pos,current_rfoot_pos,current_lfoot_dir,current_rfoot_dir,next_target_pos,next_foot_to_move):
    pass
    '''
    reward = distance_reward + constraint_reward + step_reward
    distance_reward：减少到目标点的欧氏距离的奖励（例如，Δdistance * scale）。
    constraint_reward：约束违反惩罚（如夹角超标罚-10，转向角超标罚-10）。
    step_reward：每步小惩罚（-0.1）鼓励高效步数。
    成功到达终点奖励+100。
    '''
'''
2.3.1 计算奖励值
'''
def Cul_reward(partial_map,current_bot_center_pos,current_bot_center_dir,current_lfoot_pos,current_rfoot_pos,current_lfoot_dir,current_rfoot_dir,next_target_pos,next_foot_to_move):
    pass

'''
2.3.2 NN架构
'''

Net=torch.nn.Sequential(

)

'''
3. 地形法向量计算
'''
def compute_terrain_normal(I, current_foot_pos, foot_size=(5, 3)):
    pass

'''
4. 转向角计算
'''
def compute_turning_angle(current_bot_center_pos,next_bot_center_pos,current_bot_center_dir):
    pass


'''
我们约定：

1.
所有在神经网络输入输出中使用的坐标、位移等，都需要归一化。

2.
所有在物理约束检查（如距离、角度）中使用的坐标，必须是原始尺度（像素坐标）。

因此，在函数设计时，我们明确：

•
全局路径规划（A*）和局部落足点规划（Partial_footstep_planning）中，使用的坐标都是原始尺度（像素坐标）。

•
强化学习模块（Reinforce_learning_config）的输入需要归一化，输出也是归一化的相对位移（然后需要反归一化回原始尺度）。
'''
