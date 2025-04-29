"""
Library Version:
python 3.7.9
tensorflow 2.2.0
gym 0.8.0
"""
import numpy as np
import ax as ax
import matplotlib.pyplot as plt
import matplotlib
# del matplotlib.font_manager.weight_dict['roman']  # 绘图大小显示不正常时加入此两行代码（运行一次即可）
# matplotlib.font_manager._rebuild()

# 绘图标准
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
matplotlib.rcParams['pdf.fonttype'] = 42    # 解决 PDF 论文中 python 画图 Type 3 fonts 字体不兼容问题
matplotlib.rcParams['ps.fonttype'] = 42     # 解决 PDF 论文中 python 画图 Type 3 fonts 字体不兼容问题
font = {'family': 'Times New Roman',
         'style': 'italic',
         'weight': 'normal',
         'color':  'black',
         'size': 40
        }
font_1 = {
    'family': 'Times New Roman',
    'style': 'italic',
    'weight': 'normal',
    'size': 40
}


####################  运动学变量定义  ####################
pi = 3.14159265358
PHI = 1.34                                  # 第一、二关节相位差
f = 1                                       # 关节摆动频率

theta_10f = 0.1179                          # 第一关节转动角度幅值
theta_21f = 0.3344                          # 第二关节相对于第一关节转动角度幅值

## PID设定值
theta_10_g = 0                              # PID设定的第一关节角度
theta_21_g = 0                              # PID设定的第二关节相对于第一关节角度
theta_20_g = 0                              # PID设定的第二关节角度

## 前进位移、两关节角度
X_0 = 0                                     # 前进位移
theta_10 = 0                                # 第一关节转动角度
theta_21 = theta_21f * np.sin(PHI)          # 第二关节相对于第一关节转动角度
theta_20 = theta_10 + theta_21f * np.sin(PHI)   # 第二关节转动角度


X_0_1o = 0                                  # 前进速度
U = X_0_1o                                  # 前进速度记为U，用于后续计算
theta_10_1o = 0                             # 第一关节角速度
theta_21_1o = 0                             # 第二关节相对于第一关节角速度
theta_20_1o = 0                             # 第二关节角速度

## 前进加速度、两关节角加速度
X_0_2o = 0                                  # 前进加速度
theta_10_2o = 0                             # 第一关节角加速度
theta_21_2o = 0                             # 第二关节相对于第一关节角加速度

## 速度相关变量
V_c1c = 0                                   # 第一关节质心处垂直于鱼体表面的线速度分量
V_c1 = 0                                    # 第一关节部分质心处线速度
V_c1_v = np.array([0, 0])[np.newaxis, :]
                                            # 第一关节部分质心处线速度（矢量）
V_c2c = 0                                   # 第二关节质心的垂直于鱼尾表面的线速度分量
V_c2 = 0                                    # 第二关节质心处线速度
V_c2_v = np.array([0, 0])[np.newaxis, :]    # 第二关节质心处线速度（矢量）


####################  动力学变量定义  ####################
rou = 1000                                  # 流体密度
C_0 = 0.9                                   # 头部阻力系数
C_1 = 1.54                                  # 第一关节阻力系数
C_2 = 1.54                                  # 第二关节阻力系数

Q_1 = 0                                     # 广义坐标 theta_10 对应的非保守力
Q_2 = 0                                     # 广义坐标 theta_21 对应的非保守力
Q_3 = 0                                     # 广义坐标 X_0 对应的非保守力

F_0 = 0                                     # 头部水动力
F_0_v = np.array([0, 0])[np.newaxis, :]     # 头部水动力（矢量）

F_1_s = 0                                   # 第一关节水动力
F_1_v = np.array([0, 0])[np.newaxis, :]     # 第一关节水动力（矢量）
F_1x = 0                                    # 第一关节水动力在 X 轴上的分量

F_2_s = 0                                   # 第二关节水动力
F_2_v = np.array([0, 0])[np.newaxis, :]     # 第二关节水动力（矢量）
F_2x = 0                                    # 第二关节水动力在 X 轴上的分量

## 转矩、功率、功、效率
M_1 = 0                                     # 第一个关节转矩
M_2 = 0                                     # 第二个关节转矩
P_1 = 0                                     # 第一个关节功率
P_2 = 0                                     # 第二个关节功率
P_useful = 0                                # 有用功率
W_total = 0                                 # 鱼体游动时消耗总功
W_useful = 0                                # 鱼体游动时的有用功
total_eta = 0                               # 机器鱼的推进效率


####################  PID 参数定义  ####################
## 实验时分别改变此 Kp_1 和 Kp_2，记录推进效率，详见论文
Kp_1 = 10
err_1 = 0

Kp_2 = 50
err_2 = 0


####################  机器鱼相关参数定义  ####################
m_0 = 1.2                                   # 头部质量
m_1 = 2.4                                   # 第一关节质量
m_2 = 0.6                                   # 第二关节质量

A_0x = 0.03                                 # 头部与速度方向垂直的截面积
A_1 = 0.05                                  # 第一关节截面积
A_2 = 0.05                                  # 第二关节截面积

L_0 = 0.2                                   # 头部长度
L_1 = 0.24                                  # 第一关节长度
L_c1 = 0.12                                 # 第一关节质心距离第一个关节前端的长度
L_d1 = 0.12                                 # 第一关节水动力作用点距离第一个关节前端的长度
L_2 = 0.16                                  # 第二关节长度
L_c2 = 0.08                                 # 第二关节质心距离第二个关节前端的长度
L_d2 = 0.08                                 # 第二关节水动力作用点距离第二个关节前端的长度

J_c1 = m_1 * L_1 * L_1 / 12                 # 第一关节以其质心为参考点的转动惯量
J_c2 = m_2 * L_2 * L_2 / 12                 # 第二关节以其质心为参考点的转动惯量


####################  拉格朗日方程系数变量  ####################
D_11 = 0                                    # 拉格朗日系数 D_11
D_12 = 0                                    # 拉格朗日系数 D_12
D_13 = 0                                    # 拉格朗日系数 D_13
D_21 = 0                                    # 拉格朗日系数 D_21
D_22 = 0                                    # 拉格朗日系数 D_22
D_23 = 0                                    # 拉格朗日系数 D_23
D_31 = 0                                    # 拉格朗日系数 D_31
D_32 = 0                                    # 拉格朗日系数 D_32
D_33 = 0                                    # 拉格朗日系数 D_33

D_111 = 0                                   # 拉格朗日系数 D_111
D_122 = 0                                   # 拉格朗日系数 D_122
D_211 = 0                                   # 拉格朗日系数 D_211
D_222 = 0                                   # 拉格朗日系数 D_222
D_311 = 0                                   # 拉格朗日系数 D_311
D_322 = 0                                   # 拉格朗日系数 D_322

D_112 = 0                                   # 拉格朗日系数 D_112
D_212 = 0                                   # 拉格朗日系数 D_212
D_312 = 0                                   # 拉格朗日系数 D_312

DD_1 = np.zeros((3,3))                      # 3X3 惯量矩阵
DD_2 = np.zeros((3, 2))                     # 3X2 向心项系数矩阵
DD_3 = np.zeros((3, 1))                     # 3X1 科氏力系数矩阵
g_acc = np.zeros((3, 1))                    # 广义加速度


####################  其他变量记录  ####################
time = 30                                   # 仿真时间
dt = 0.01                                   # 仿真时间间隔
counter = 0                                 # 计算循环次数

## 记录两关节受力、线速度
F_1_s_t = []                                # 第一关节受力大小，标量
F_1_x_t = []                                # 第一关节受力，X 轴分量
F_1_y_t = []                                # 第一关节受力，Y 轴分量
V_c1c_t = []                                # 第一关节中心线速度
F_2_s_t = []                                # 第二关节受力大小，标量
F_2_x_t = []                                # 第二关节受力，X 轴分量
F_2_y_t = []                                # 第二关节受力，Y 轴分量
V_c2c_t = []                                # 第二关节中心线速度

## 记录PID设定的角度信号
theta_10_g_t = []
theta_21_g_t = []
theta_20_g_t = []

## 记录前进位移、两关节角度
X_0_t = []                                  # 头部前进位移
theta_10_t = []                             # 第一关节转动角度
theta_21_t = []                             # 第二关节相对于第一关节转动角度
theta_20_t = []                             # 第二关节转动角度

##记录误差
error = []

## 记录前进速度、两关节角速度
X_0_1o_t = []                               # 头部前进速度
theta_10_1o_t = []                          # 第一关节角速度
theta_21_1o_t = []                          # 第二关节相对于第一关节角速度
theta_20_1o_t = []                          # 第二关节角速度

## 记录前进加速度、两关节角加速度
X_0_2o_t = []                               # 头部前进加速度
theta_10_2o_t = []                          # 第一关节角加速度
theta_21_2o_t = []                          # 第二关节相对于第一关节角加速度
theta_20_2o_t = []                          # 第二关节角加速度

## 记录两关节的转矩、功率及其他
M_1_t = []                                  # 第一关节转矩
M_2_t = []                                  # 第二关节转矩
P_1_t = []                                  # 第一关节功率
P_2_t = []                                  # 第二关节功率
P_useful_t = []                             # 有用功率
total_eta_t = []                            # 推进效率


####################  开始运算  ####################
for t in np.arange(dt, time, dt):           # 从0.01秒到结束
    counter = counter + 1                   # 计数器加1

    theta_10_g = theta_10f * np.sin(2 * pi * f * t)         # PID 设定的第一关节随时间变化的角度
    theta_21_g = theta_21f * np.sin(2 * pi * f * t + PHI)   # PID 设定的第二关节相对于第一关节随时间变化的角度
    theta_20_g = theta_10_g + theta_21_g                    # PID 设定的第二关节随时间变化的角度
    theta_10_g_t.append(theta_10_g)         # 记录到列表（下同）
    theta_21_g_t.append(theta_21_g)
    theta_20_g_t.append(theta_20_g)

    err_1 = theta_10_g - theta_10           # 第一个关节设定角度与实际角度之间差值
    err_2 = theta_21_g - theta_21           # 第二个关节设定角度与实际角度之间差值（相对值）
    error.append(err_1)

    M_1 = Kp_1 * err_1                      # 第一个关节的转矩
    M_2 = Kp_2 * err_2                      # 第二个关节的转矩
    M_1_t.append(M_1)
    M_2_t.append(M_2)
    # print(M_1, M_2)

    P_1 = M_1 * theta_10_1o                 # 第一个关节消耗总功率：转矩×角速度
    P_2 = M_2 * theta_21_1o                 # 第二个关节消耗总功率：转矩×角速度
    # print(P_1,P_2)

    # 拉格朗日各系数表达式
    D_11 = J_c1 + J_c2 + m_1 * L_c1 ** 2 + m_2 * (L_1 ** 2 + L_c2 ** 2 + 2 * L_1 * L_c2 * np.cos(theta_21))
    D_12 = J_c2 + m_2 * (L_c2 ** 2 + L_1 * L_c2 * np.cos(theta_21))
    D_21 = D_12
    D_13 = -m_1 * L_c1 * np.sin(theta_10) - m_2 * (L_1 * np.sin(theta_10) + L_c2 * np.sin(theta_20))
    D_31 = D_13
    D_22 = m_2 * L_c2 ** 2 + J_c2
    D_23 = -m_2 * L_c2 * np.sin(theta_20)
    D_32 = D_23
    D_33 = m_0 + m_1 + m_2
    D_111 = 0
    D_222 = 0
    D_122 = -m_2 * L_1 * L_c2 * np.sin(theta_21)
    D_211 = -D_122
    D_311 = -m_1 * L_c1 * np.cos(theta_10) - m_2 * (L_1 * np.cos(theta_10) + L_c2 * np.cos(theta_20))
    D_322 = -m_2 * L_c2 * np.cos(theta_20)
    D_112 = -2 * m_2 * L_1 * L_c2 * np.sin(theta_21)
    D_212 = 0
    D_312 = -2 * m_2 * L_c2 * np.cos(theta_20)

    # 惯量矩阵
    DD_1 = np.array([[D_11, D_12, D_13],
                     [D_21, D_22, D_23],
                     [D_31, D_32, D_33]])
    # 向心项系数矩阵
    DD_2 = np.array([[D_111, D_122],
                     [D_211, D_222],
                     [D_311, D_322]])
    # 科氏力系数矩阵
    DD_3 = np.array([[D_112],
                     [D_212],
                     [D_312]])

    # 第一关节速度V_c1矢量
    V_c1_v = np.array([X_0_1o - L_1 * theta_10_1o * np.sin(theta_10),
                       L_1 * theta_10_1o * np.cos(theta_10)])[np.newaxis,:]
    # 第一关节质心处单位法向量
    I_1 = np.array([[np.sin(theta_10)], [-np.cos(theta_10)]])
    # 第一关节质心处法向速度
    V_c1c = np.dot(V_c1_v, I_1)
    V_c1c_t.append(V_c1c[0,0])

    # 第一关节受力
    if V_c1c[0,0] > 0:  # 当摆动方向和法向量方向相同时，身体部分水动力
        F_1_v = np.array([-0.5 * rou * C_1 * V_c1c[0,0] ** 2 * A_1 * np.sin(theta_10),
                          0.5 * rou * C_1 * V_c1c[0,0] ** 2 * A_1 * np.cos(theta_10)])[np.newaxis,:]
        F_1_x_t.append(F_1_v[0, 0])
        F_1_y_t.append(F_1_v[0, 1])
        F_1x = F_1_v[0, 0]
    else:
        F_1_v = np.array([0.5 * rou * C_1 * V_c1c[0,0] ** 2 * A_1 * np.sin(theta_10),
                          -0.5 * rou * C_1 * V_c1c[0,0] ** 2 * A_1 * np.cos(theta_10)])[np.newaxis,:]
        F_1_x_t.append(F_1_v[0, 0])
        F_1_y_t.append(F_1_v[0, 1])
        F_1x = F_1_v[0, 0]


    # 第二关节速度V_c2矢量
    V_c2_v = np.array([X_0_1o - L_1 * theta_10_1o * np.sin(theta_10) - L_c2 * theta_20_1o * np.sin(theta_20),
                       L_1 * theta_10_1o * np.cos(theta_10) + L_c2 * theta_20_1o * np.cos(theta_20)])[np.newaxis,:]
    # 第二关节质心处单位法向量
    I_2 = np.array([[np.sin(theta_20)], [-np.cos(theta_20)]])
    # 第一关节质心处法向速度
    V_c2c = np.dot(V_c2_v, I_2)
    V_c2c_t.append(V_c2c[0, 0])

    # 第二关节受力
    if V_c2c[0,0] > 0:  # 当摆动方向和法向量方向相同时，尾部水动力
        F_2_v = np.array([-0.5 * rou * C_2 * V_c2c[0,0] ** 2 * A_2 * np.sin(theta_20),
                          0.5 * rou * C_2 * V_c2c[0,0] ** 2 * A_2 * np.cos(theta_20)])[np.newaxis,:]
        F_2_x_t.append(F_2_v[0, 0])
        F_2_y_t.append(F_2_v[0, 1])
        F_2x = F_2_v[0, 0]
    else:
        F_2_v = np.array([0.5 * rou * C_2 * V_c2c[0,0] ** 2 * A_2 * np.sin(theta_20),
                          -0.5 * rou * C_2 * V_c2c[0,0] ** 2 * A_2 * np.cos(theta_20)])[np.newaxis,:]
        F_2_x_t.append(F_2_v[0, 0])
        F_2_y_t.append(F_2_v[0, 1])
        F_2x = F_2_v[0, 0]

    # 头部受力情况
    if X_0_1o >= 0:
        F_0 = -0.5 * rou * C_0 * X_0_1o ** 2 * A_0x
    else:
        F_0 = 0.5 * rou * C_0 * X_0_1o ** 2 * A_0x
    F_0_v = np.array([F_0, 0])[np.newaxis,:]
    # print(F_2x, F_1x, F_0)

    # 非保守力的求解
    Q_1 = np.dot(F_1_v, np.array([[-L_d1 * np.sin(theta_10)], [L_d1 * np.cos(theta_10)]]))[0, 0] + np.dot(F_2_v, np.array([[-L_1 * np.sin(theta_10) - L_d2 * np.sin(theta_20)], [L_1 * np.cos(theta_10) + L_d2 * np.cos(theta_20)]]))[0, 0] + M_1
    Q_2 = np.dot(F_2_v, np.array([[-L_d2 * np.sin(theta_20)], [L_d2 * np.cos(theta_20)]]))[0, 0] + M_2
    Q_3 = np.dot(F_0_v, np.array([[1], [0]]))[0, 0] + np.dot(F_1_v, np.array([[1], [0]]))[0, 0] + np.dot(F_2_v, np.array([[1], [0]]))[0, 0]

    # 拉格朗日方程逆解求广义加速度
    g_acc = np.dot(np.linalg.inv(DD_1), (np.array([[Q_1], [Q_2], [Q_3]]) - np.dot(DD_2, np.array([[theta_10_1o ** 2], [theta_21_1o ** 2]])) - DD_3 * (theta_10_1o * theta_21_1o)))

    # 两关节角加速度、前进加速度
    theta_10_2o = g_acc[0, 0]               # 第一关节角加速度
    theta_21_2o = g_acc[1, 0]               # 第二关节相对角加速度
    X_0_2o = g_acc[2, 0]                    # 前进加速度
    theta_10_2o_t.append(theta_10_2o)
    theta_21_2o_t.append(theta_21_2o)
    X_0_2o_t.append(X_0_2o)

    # 两关节角速度、前进速度
    theta_10_1o = theta_10_1o + theta_10_2o * dt    # 第一关节角速度
    theta_21_1o = theta_21_1o + theta_21_2o * dt    # 第二关节相对角速度
    theta_20_1o = theta_10_1o + theta_21_1o         # 第二关节角速度
    X_0_1o = X_0_1o + X_0_2o * dt                   # 前进速度
    U = 7 * X_0_1o
    theta_10_1o_t.append(theta_10_1o)
    theta_21_1o_t.append(theta_21_1o)
    theta_20_1o_t.append(theta_20_1o)
    X_0_1o_t.append(X_0_1o)

    # 两关节角度
    theta_10 = theta_10 + theta_10_1o * dt  # 第一关节转动角度
    theta_21 = theta_21 + theta_21_1o * dt  # 第二关节相对转动角度
    theta_20 = theta_10 + theta_21          # 第二关节转动角度
    theta_10_t.append(theta_10)
    theta_21_t.append(theta_21)
    theta_20_t.append(theta_20)

    # 有用推进功率
    P_useful = (F_1x + F_2x) * U
    P_useful_t.append(P_useful)

    # 积分计算有用功
    W_useful = W_useful + P_useful * dt
    W_total = W_total + (P_1 + P_2) * dt

    # 计算效率
    if W_total == 0:
        total_eta = 0
    else:
        total_eta = W_useful / W_total
    if total_eta > 1 or total_eta < 0:
        total_eta = 0
    total_eta_t.append(100*total_eta)


    ####################  数据保存（不要重复保存）  ####################
    # 每隔一段时间保存效率到txt，用于后续绘图对比
    # if (counter % 30) == 29:
    #     my_file = open('PID_eta.txt', 'a')
    #     text = 'ep_r: %.1f\n' % (100*total_eta)
    #     my_file.write(text)
    #     my_file.close()

    # 保存角度、角速度等变量数据到txt，用于后续绘图对比
    # my_file = open('PID_state.txt', 'a')
    # text = 'Speed: %f, angle1: %f, angle2: %f, v1: %f, v2: %f, M1: %f, M2: %f\n' % (X_0_1o, theta_10, theta_21, theta_10_1o, theta_21_1o, M_1, M_2)
    # my_file.write(text)
    # my_file.close()


####################  输出结果  ####################
print('W_total = ', W_total)
print('W_useful = ', W_useful)
print('Total_eta = ', total_eta)


####################  绘制图像  ####################
# 绘制推进速度图
# figsize = 15, 10
# figure, ax = plt.subplots(figsize=figsize)
# x = range(0, 3500, 500)
# plt.xticks(x, ('0','5','10','15','20','25','30'))
# plt.plot(X_0_1o_t, linewidth=2, label='Travelling Speed')
# plt.tick_params(labelsize=40)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.ylabel('speed(m/s)',font)
# plt.xlabel('time(s)',font)
# plt.legend(loc='best', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
# plt.savefig('./Figure1_speed.pdf')
# plt.savefig('./Figure1_speed.png')


# 绘制关节角度图
# fig = plt.figure(2, figsize=(15, 10))
# ax1 = fig.add_subplot(211)
# x = range(0, 3500, 500)
# plt.xticks(x, ('0','5','10','15','20','25','30'))
# plt.ylim(-0.6, 0.6)
# plt.xlim(-100, 3100)
# ax1.plot(theta_10_t, linewidth=2, label='1st joint')
# ax1.set_ylabel('angle(rad)',font)
# plt.tick_params(labelsize=40)
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
#
# ax2 = fig.add_subplot(212)
# x = range(0, 3500, 500)
# plt.xticks(x, ('0','5','10','15','20','25','30'))
# plt.ylim(-0.6, 0.6)
# plt.xlim(-100, 3100)
# ax2.plot(theta_21_t, linewidth=2, label='2nd joint', color='orange')
# ax2.set_xlabel('time(s)',font)
# plt.tick_params(labelsize=40)
# labels = ax2.get_xticklabels() + ax2.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
# plt.savefig('./Figure2_angle.pdf')
# plt.savefig('./Figure2_angle.png')


# 绘制关节角速度图
# fig = plt.figure(3, figsize=(15, 10))
# ax1 = fig.add_subplot(211)
# x = range(0, 3500, 500)
# plt.xticks(x, ('0','5','10','15','20','25','30'))
# plt.ylim(-4, 4)
# plt.xlim(-100, 3100)
# ax1.plot(theta_10_1o_t, linewidth=2, label='1st joint')
# ax1.set_ylabel('angle velocity(rad/s)',font)
# plt.tick_params(labelsize=40)
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
#
# ax2 = fig.add_subplot(212)
# x = range(0, 3500, 500)
# plt.xticks(x, ('0','5','10','15','20','25','30'))
# plt.ylim(-4, 4)
# plt.xlim(-100, 3100)
# ax2.plot(theta_21_1o_t, linewidth=2, label='2nd joint', color='orange')
# ax2.set_xlabel('time(s)',font)
# plt.tick_params(labelsize=40)
# labels = ax2.get_xticklabels() + ax2.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
# plt.savefig('./Figure3_anglevelocity.pdf')
# plt.savefig('./Figure3_anglevelocity.png')


# 绘制关节加速度图
# fig = plt.figure(4, figsize=(15, 10))
# ax1 = fig.add_subplot(211)
# x = range(0, 3500, 500)
# plt.xticks(x, ('0','5','10','15','20','25','30'))
# # plt.ylim(-0.3, 0.3)
# plt.xlim(-100, 3100)
# ax1.plot(theta_10_2o_t, linewidth=2, label='1st joint')
# ax1.set_ylabel('angle acceleration(rad/s^2)',font)
# plt.tick_params(labelsize=40)
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
#
# ax2 = fig.add_subplot(212)
# x = range(0, 3500, 500)
# plt.xticks(x, ('0','5','10','15','20','25','30'))
# # plt.ylim(-0.5, 0.5)
# plt.xlim(-100, 3100)
# ax2.plot(theta_21_2o_t, linewidth=2, label='2nd joint', color='orange')
# ax2.set_xlabel('time(s)',font)
# plt.tick_params(labelsize=40)
# labels = ax2.get_xticklabels() + ax2.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
# plt.savefig('./Figure4_acceleration.pdf')
# plt.savefig('./Figure4_acceleration.png')


# 绘制关节转矩图
# fig = plt.figure(5, figsize=(15, 10))
# ax1 = fig.add_subplot(211)
# x = range(0, 3500, 500)
# plt.xticks(x, ('0','5','10','15','20','25','30'))
# plt.ylim(-3, 3)
# plt.xlim(-100, 3100)
# ax1.plot(M_1_t, linewidth=2, label='1st joint')
# ax1.set_ylabel('torque',font)
# plt.tick_params(labelsize=40)
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
#
# ax2 = fig.add_subplot(212)
# x = range(0, 3500, 500)
# plt.xticks(x, ('0','5','10','15','20','25','30'))
# plt.ylim(-0.6, 0.6)
# plt.xlim(-100, 3100)
# ax2.plot(M_2_t, linewidth=2, label='2nd joint', color='orange')
# ax2.set_xlabel('time(s)',font)
# plt.tick_params(labelsize=40)
# labels = ax2.get_xticklabels() + ax2.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
# plt.savefig('./Figure5_torque.pdf')
# plt.savefig('./Figure5_torque.png')


# 绘制效率图
fig = plt.figure(6, figsize=(15, 10))
ax = fig.add_subplot(111)
x = range(0, 3500, 500)
plt.xticks(x, ('0','5','10','15','20','25','30'))
plt.ylim(-3, 60)
plt.xlim(-100, 3100)
ax.plot(total_eta_t, linewidth=2, label='Propulsion Efficiency')
ax.set_xlabel('time(s)',font)
ax.set_ylabel('%',font)
plt.tick_params(labelsize=40)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
# plt.savefig('./Figure6_eta.pdf')
# plt.savefig('./Figure6_eta.png')

plt.show()

