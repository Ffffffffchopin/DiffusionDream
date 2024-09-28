import numpy as np


# 计算鼠标位移
def calculate_mouse_movement(R):
    # 提取旋转矩阵中的元素
    r11, r12, r13 = R[0]
    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]

    # 计算绕 x 轴和 y 轴的旋转角度
    # 使用反三角函数来计算角度
    # 注意这里使用的是弧度制
    pitch = -np.arcsin(r31)  # 绕 x 轴的旋转 (pitch)
    yaw = np.arctan2(r32, r33)  # 绕 y 轴的旋转 (yaw)

    # 假设灵敏度为 1
    sensitivity = 1.0

    # 将角度转换为像素位移
    # 这里假设每个弧度对应一个单位像素
    dx = yaw * sensitivity
    dy = pitch * sensitivity

    return dx, dy




# 测试计算
if __name__ == '__main__':
    R = np.array([[ 9.998e-01 ,-9.500e-03 , 2.010e-02  ],[ 9.500e-03 , 1.000e+00 , 2.400e-03],[-2.010e-02 ,-2.200e-03 , 9.998e-01]])
    dx, dy = calculate_mouse_movement(R)
    print(f"鼠标相对位移: dx = {dx:.4f}, dy = {dy:.4f}")