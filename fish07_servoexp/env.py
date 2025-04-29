"""
Library Version:
python 3.7.9
tensorflow 2.2.0
gym 0.8.0
pyglet 2.0.5
"""
import numpy as np
import matplotlib.pyplot as plt
import pyglet
import matplotlib
import math

# 绘图文字标准
plt.rcParams['axes.unicode_minus'] = False                  # 用来正常显示负号
matplotlib.rcParams['pdf.fonttype'] = 42                    # 用来解决PDF论文中python画图 Type 3 fonts 字体不兼容问题
matplotlib.rcParams['ps.fonttype'] = 42                     # 用来解决PDF论文中python画图 Type 3 fonts 字体不兼容问题
font = {'family': 'Times New Roman',
        'style': 'italic',
        'weight': 'normal',
        'color': 'black',
        'size': 40
        }
font_1 = {
    'family': 'Times New Roman',
    'style': 'italic',
    'weight': 'normal',
    'size': 40
}

# 机器鱼模型
class FishEnv(object):
    viewer = None
    state_dim = 6
    action_dim = 2
    action_bound = [11, 1]

    def __init__(self):
        ####################  运动学变量定义  ####################
        self.pi = 3.14159265358
        self.PHI = 1.34                                     # 第一关节第二关节相位差
        self.f = 1                                          # 关节摆动频率

        self.theta_10f = 0.1179                             # 第一关节相对于头部的转动角度幅值
        self.theta_21f = 0.3344                             # 第二关节相对于第一关节转动的角度幅值

        self.goal = 0
        self.on_goal = 0

        ## PID设定值
        self.theta_10_g = 0                                 # PID设定的第一关节角度
        self.theta_21_g = 0                                 # PID设定的第二关节相对于第一关节角度
        self.theta_20_g = 0                                 # PID设定的第二关节角度

        self.theta_10_1o_g = 0
        self.theta_21_1o_g = 0
        self.theta_10_2o_g = 0
        self.theta_21_2o_g = 0


        ## 前进位移、两关节角度
        self.X_0 = 0
        self.theta_10 = 0                                   # 第一关节转动角度
        self.theta_21 = self.theta_21f * np.sin(self.PHI)   # 第二关节相对于第一关节转动角度
        self.theta_20 = self.theta_21f * np.sin(self.PHI)   # 第二关节转动角度

        ## 前进速度、两关节角速度
        self.X_0_1o = 0                                     # 前进速度
        self.U = self.X_0_1o                                # 前进速度记为 U，用于后续计算
        self.theta_10_1o = 0                                # 第一关节角速度
        self.theta_21_1o = 0                                # 第二关节相对于第一关节角速度
        self.theta_20_1o = 0                                # 第二关节角速度

        ## 前进速度、两关节角速度
        self.X_0_2o = 0                                     # 前进加速度
        self.theta_10_2o = 0                                # 第一关节角加速度
        self.theta_21_2o = 0                                # 第二关节相对于第一关节角加速度
        self.theta_20_2o = 0                                # 第二关节角加速度

        ## 速度相关变量
        self.V_c1c = 0                                      # 第一关节质心处垂直于鱼尾表面的线速度分量
        self.V_c1 = 0                                       # 第一关节质心处线速度
        self.V_c1_v = np.array([0, 0])[np.newaxis, :]       # 第一关节质心处线速度，矢量

        self.V_c2c = 0                                      # 第二关节质心处垂直于鱼尾表面的线速度分量
        self.V_c2 = 0                                       # 第二关节质心处线速度
        self.V_c2_v = np.array([0, 0])[np.newaxis, :]       # 第二关节质心处线速度，矢量

        self.I_1 = np.array([0, 0])[:, np.newaxis]          # 第一关节质心单位法向量
        self.I_2 = np.array([0, 0])[:, np.newaxis]          # 第二关节质心单位法向量


        ####################  动力学变量定义  ####################
        self.rou = 1000                                     # 流体密度
        self.C_0 = 0.9                                      # 头部阻力系数
        self.C_1 = 1.54                                     # 第一关节阻力系数
        self.C_2 = 1.54                                     # 第二关节阻力系数

        self.Q_1 = 0                                        # 广义坐标 theta_10 对应的非保守力
        self.Q_2 = 0                                        # 广义坐标 theta_21 对应的非保守力
        self.Q_3 = 0                                        # 广义坐标 X_0 对应的非保守力

        self.F_0 = 0                                        # 头部水动力
        self.F_0_v = np.array([0, 0])[np.newaxis, :]        # 头部水动力，矢量

        self.F_1_v = np.array([0, 0])[np.newaxis, :]        # 第一关节水动力（矢量）
        self.F_1x = 0                                       # 第一关节水动力在 X 轴上的分量
        self.F_2_v = np.array([0, 0])[np.newaxis, :]        # 第二关节水动力（矢量）
        self.F_2x = 0                                       # 第二关节水动力在 X 轴上的分量

        ## 转矩、功率、功、效率
        self.M_1 = 0                                        # 第一个关节的转矩
        self.M_2 = 0                                        # 第二个关节的转矩
        self.P_1 = 0                                        # 第一个关节的的功率
        self.P_2 = 0                                        # 第二个关节的功率
        self.P_useful = 0                                   # 有用功率
        self.W_total = 0                                    # 鱼体游动时消耗的总功
        self.W_useful = 0                                   # 鱼体游动时的有用功
        self.total_eta = 0                                  # 机器鱼的推进效率
        self.reward = 0                                     # 奖励


        ####################  PID 参数定义  ####################
        self.err_1 = 0
        self.err_2 = 0


        ####################  机器鱼相关参数定义  ####################
        self.m_0 = 1.2                                      # 头部质量
        self.m_1 = 2.4                                      # 第一关节质量
        self.m_2 = 0.6                                      # 第二关节质量

        self.A_0x = 0.03                                    # 头部与速度方向垂直面积
        self.A_1 = 0.05                                     # 第一关节截面积
        self.A_2 = 0.05                                     # 第二关节截面积

        self.L_0 = 0.2                                      # 头部长度
        self.L_1 = 0.24                                     # 第一关节长度
        self.L_c1 = 0.12                                    # 第一关节质心距离第一个关节前端的长度
        self.L_d1 = 0.12                                    # 第一关节水动力作用点距离第一个关节前端的长度
        self.L_2 = 0.16                                     # 第二关节长度
        self.L_c2 = 0.08                                    # 第二关节质心距离第二个关节前端的长度
        self.L_d2 = 0.08                                    # 第二关节水动力作用点距离第二个关节前端的长度

        self.J_c1 = self.m_1 * self.L_1 * self.L_1 / 12     # 第一关节以其质心为参考点的转动惯量
        self.J_c2 = self.m_2 * self.L_2 * self.L_2 / 12     # 第二关节以其质心为参考点的转动惯量


        ####################  拉格朗日方程系数变量  ####################
        self.D_11 = 0                                       # 拉格朗日系数 D_11
        self.D_12 = 0                                       # 拉格朗日系数 D_12
        self.D_13 = 0                                       # 拉格朗日系数 D_13
        self.D_21 = 0                                       # 拉格朗日系数 D_21
        self.D_22 = 0                                       # 拉格朗日系数 D_22
        self.D_23 = 0                                       # 拉格朗日系数 D_23
        self.D_31 = 0                                       # 拉格朗日系数 D_31
        self.D_32 = 0                                       # 拉格朗日系数 D_32
        self.D_33 = 0                                       # 拉格朗日系数 D_33

        self.D_111 = 0                                      # 拉格朗日系数 D_111
        self.D_122 = 0                                      # 拉格朗日系数 D_122
        self.D_211 = 0                                      # 拉格朗日系数 D_211
        self.D_222 = 0                                      # 拉格朗日系数 D_222
        self.D_311 = 0                                      # 拉格朗日系数 D_311
        self.D_322 = 0                                      # 拉格朗日系数 D_322

        self.D_112 = 0                                      # 拉格朗日系数 D_112
        self.D_212 = 0                                      # 拉格朗日系数 D_212
        self.D_312 = 0                                      # 拉格朗日系数 D_312

        self.DD_1 = np.zeros((3, 3))                        # 3X3 惯量矩阵
        self.DD_2 = np.zeros((3, 2))                        # 3X2 向心项系数矩阵
        self.DD_3 = np.zeros((3, 1))                        # 3X1 科氏力系数矩阵
        self.DD_1_inv = np.zeros((3, 3))                    # DD_1 矩阵的逆
        self.g_acc = np.zeros((3, 1))                       # 广义加速度

        ####################  其他变量记录  ####################
        self.dt = 0.01                                      # 仿真时间间隔
        self.counter = 0                                    # 计算循环次数，将数据存入到相应数组中的元素
        # self.x_axis = np.zeros((1, 3000000))                # 画图用X轴坐标，与时间轴上取点个数相同
        self.x_axis = []

        # 记录PID设定角度
        self.theta_10_g_t = []
        self.theta_21_g_t = []

        ## 记录角度
        self.theta_10_t = []
        self.theta_21_t = []

        ##记录误差
        self.error = []

        # 记录前进速度、两关节角速度
        self.X_0_1o_t = []                                  # 前进速度
        self.theta_10_1o_t = []                             # 第一关节角速度
        self.theta_21_1o_t = []                             # 第二关节相对于第一关节角速度
        self.theta_20_1o_t = []                             # 第二关节角速度

        # 记录前进加速度、两关节角加速度
        self.X_0_2o_t = []                                  # 前进加速度
        self.theta_10_2o_t = []                             # 第一关节角加速度
        self.theta_21_2o_t = []                             # 第二关节相对于第一关节角加速度
        self.theta_20_2o_t = []                             # 第二关节角加速度

        ## 记录转矩及其他
        self.M_1_t = []
        self.M_2_t = []
        self.action0 = []
        self.action1 = []

        self.eta_t = []
        # self.eta_t = np.zeros((1, int(3000000)))

        self.fish_info = np.zeros(6, dtype=[('r', np.float32), ])
        self.a = np.zeros(2)


    ####################  开始运算  ####################
    def step(self, action):
        done = False
        self.counter = self.counter + 1                     # 计数器加1
        self.x_axis.append(self.counter)

        self.theta_10_g = self.theta_10f * np.sin(2 * self.pi * self.f * self.dt * self.counter)
        self.theta_21_g = self.theta_21f * np.sin(2 * self.pi * self.f * self.dt * self.counter + self.PHI)
        self.theta_10_1o_g = self.theta_10f * 2 * self.pi * self.f * np.cos(2 * self.pi * self.f * self.dt * self.counter)
        self.theta_21_1o_g = self.theta_21f * 2 * self.pi * self.f * np.cos(2 * self.pi * self.f * self.dt * self.counter + self.PHI)
        self.theta_10_2o_g = -self.theta_10f * 2 * self.pi * self.f * 2 * self.pi * self.f * np.sin(2 * self.pi * self.f * self.dt * self.counter)
        self.theta_21_2o_g = -self.theta_21f * 2 * self.pi * self.f * 2 * self.pi * self.f * np.sin(2 * self.pi * self.f * self.dt * self.counter + self.PHI)


        self.theta_20_g = self.theta_10_g + self.theta_21_g
        self.theta_10_g_t.append(self.theta_10_g)
        self.theta_21_g_t.append(self.theta_21_g)

        self.err_1 = self.theta_10_g - self.theta_10        # 第一个关节当前角度与设定角度之间差值
        self.err_2 = self.theta_21_g - self.theta_21        # 第二个关节当前角度与设定角度之间差值
        
        self.error.append(self.err_1)

        self.M_1 = action[0] * self.err_1                   # 第一个关节的转矩
        self.M_2 = action[1] * self.err_2                   # 第二个关节的转矩
        self.M_1_t.append(self.M_1)
        self.M_2_t.append(self.M_2)
        self.action0.append(action[0])
        self.action1.append(action[1])

        self.P_1 = self.M_1 * self.theta_10_1o              # 第一个关节的功率
        self.P_2 = self.M_2 * self.theta_21_1o              # 第二个关节的功率

        ## 拉格朗日系数表达式
        self.D_11 = self.J_c1 + self.J_c2 + self.m_1 * self.L_c1 * self.L_c1 + self.m_2 * (
                    self.L_1 * self.L_1 + self.L_c2 * self.L_c2 + 2 * self.L_1 * self.L_c2 * np.cos(self.theta_21))
        self.D_12 = self.J_c2 + self.m_2 * (self.L_c2 * self.L_c2 + self.L_1 * self.L_c2 * np.cos(self.theta_21))
        self.D_21 = self.D_12
        self.D_13 = -self.m_1 * self.L_c1 * np.sin(self.theta_10) - self.m_2 * (
                    self.L_1 * np.sin(self.theta_10) + self.L_c2 * np.sin(self.theta_20))
        self.D_31 = self.D_13
        self.D_22 = self.m_2 * self.L_c2 * self.L_c2 + self.J_c2
        self.D_23 = -self.m_2 * self.L_c2 * np.sin(self.theta_20)
        self.D_32 = self.D_23
        self.D_33 = self.m_0 + self.m_1 + self.m_2
        self.D_111 = 0
        self.D_222 = 0
        self.D_122 = -self.m_2 * self.L_1 * self.L_c2 * np.sin(self.theta_21)
        self.D_211 = -self.D_122
        self.D_311 = -self.m_1 * self.L_c1 * np.cos(self.theta_10) - self.m_2 * (
                    self.L_1 * np.cos(self.theta_10) + self.L_c2 * np.cos(self.theta_20))
        self.D_322 = -self.m_2 * self.L_c2 * np.cos(self.theta_20)
        self.D_112 = -2 * self.m_2 * self.L_1 * self.L_c2 * np.sin(self.theta_21)
        self.D_212 = 0
        self.D_312 = -2 * self.m_2 * self.L_c2 * np.cos(self.theta_20)

        ## 惯量矩阵
        self.DD_1 = np.array([[self.D_11, self.D_12, self.D_13],
                              [self.D_21, self.D_22, self.D_23],
                              [self.D_31, self.D_32, self.D_33]])
        ## 向心项系数矩阵
        self.DD_2 = np.array([[self.D_111, self.D_122],
                              [self.D_211, self.D_222],
                              [self.D_311, self.D_322]])
        ## 科氏力系数矩阵
        self.DD_3 = np.array([[self.D_112],
                              [self.D_212],
                              [self.D_312]])

        # 第一关节速度V_c1矢量
        self.V_c1_v = np.array([self.X_0_1o - self.L_1 * np.sin(self.theta_10) * self.theta_10_1o,
                                self.L_1 * np.cos(self.theta_10) * self.theta_10_1o])[np.newaxis, :]
        # 第一关节质心处单位法向量
        self.I_1 = np.array([[np.sin(self.theta_10)], [-np.cos(self.theta_10)]])
        # 第一关节质心处法向速度
        self.V_c1c = np.dot(self.V_c1_v, self.I_1)

        # 第一关节受力
        if self.V_c1c[0, 0] > 0:    # 当摆动方向和法向量方向相同时，身体部分产生的水动力
            self.F_1_v = np.array([-0.5 * self.rou * self.C_1 * self.V_c1c[0, 0] ** 2 * self.A_1 * np.sin(self.theta_10),
                                   0.5 * self.rou * self.C_1 * self.V_c1c[0, 0] ** 2 * self.A_1 * np.cos(self.theta_10)])[np.newaxis, :]
            self.F_1x = self.F_1_v[0, 0]
        else:
            self.F_1_v = np.array([0.5 * self.rou * self.C_1 * self.V_c1c[0, 0] ** 2 * self.A_1 * np.sin(self.theta_10),
                                   -0.5 * self.rou * self.C_1 * self.V_c1c[0, 0] ** 2 * self.A_1 * np.cos(self.theta_10)])[np.newaxis, :]
            self.F_1x = self.F_1_v[0, 0]


        # 第二关节速度V_c2矢量
        self.V_c2_v = np.array([self.X_0_1o - self.L_1 * np.sin(self.theta_10) * self.theta_10_1o - self.L_c2 * np.sin(self.theta_20) * self.theta_20_1o,
                                self.L_1 * np.cos(self.theta_10) * self.theta_10_1o + self.L_c2 * np.cos(self.theta_20) * self.theta_20_1o])[np.newaxis, :]
        # 第二关节质心处单位法向量
        self.I_2 = np.array([[np.sin(self.theta_20)], [-np.cos(self.theta_20)]])
        # 第一关节质心处法向速度
        self.V_c2c = np.dot(self.V_c2_v, self.I_2)

        # 第二关节受力
        if self.V_c2c[0, 0] > 0:    # 当摆动方向和法向量方向相同时，尾部产生的水动力
            self.F_2_v = np.array([-0.5 * self.rou * self.C_2 * self.V_c2c[0, 0] ** 2 * self.A_2 * np.sin(self.theta_20),
                                   0.5 * self.rou * self.C_2 * self.V_c2c[0, 0] ** 2 * self.A_2 * np.cos(self.theta_20)])[np.newaxis, :]
            self.F_2x = self.F_2_v[0, 0]
        else:
            self.F_2_v = np.array([0.5 * self.rou * self.C_2 * self.V_c2c[0, 0] ** 2 * self.A_2 * np.sin(self.theta_20),
                                   -0.5 * self.rou * self.C_2 * self.V_c2c[0, 0] ** 2 * self.A_2 * np.cos(self.theta_20)])[np.newaxis, :]
            self.F_2x = self.F_2_v[0, 0]

        # 头部受力情况
        if self.X_0_1o >= 0:
            self.F_0 = -0.5 * self.rou * self.C_0 * self.X_0_1o ** 2 * self.A_0x
        else:
            self.F_0 = 0.5 * self.rou * self.C_0 * self.X_0_1o ** 2 * self.A_0x
        self.F_0_v = np.array([self.F_0, 0])[np.newaxis, :]

        # 非保守力的求解
        self.Q_1 = np.dot(self.F_2_v, np.array([[-self.L_1 * np.sin(self.theta_10) - self.L_d2 * np.sin(self.theta_20)], [self.L_1 * np.cos(self.theta_10) + self.L_d2 * np.cos(self.theta_20)]]))[0, 0] + self.M_1
        self.Q_2 = np.dot(self.F_2_v, np.array([[-self.L_d2 * np.sin(self.theta_20)], [self.L_d2 * np.cos(self.theta_20)]]))[0, 0] + self.M_2
        self.Q_3 = np.dot(self.F_0_v, np.array([[1], [0]]))[0, 0] + np.dot(self.F_2_v, np.array([[1], [0]]))[0, 0]

        # 拉格朗日方程逆解求广义加速度
        g_acc = np.dot(np.linalg.inv(self.DD_1), (np.array([[self.Q_1], [self.Q_2], [self.Q_3]]) - np.dot(self.DD_2, np.array([[self.theta_10_1o ** 2], [self.theta_21_1o ** 2]])) - self.DD_3 * (self.theta_10_1o * self.theta_21_1o)))

        # 两关节角加速度、前进加速度
        self.theta_10_2o = g_acc[0, 0]
        self.theta_21_2o = g_acc[1, 0]
        self.X_0_2o = g_acc[2, 0]
        self.theta_10_2o_t.append(self.theta_10_2o)
        self.theta_21_2o_t.append(self.theta_21_2o)
        self.X_0_2o_t.append(self.X_0_2o)

        # 两关节角速度、前进速度
        self.theta_10_1o += self.theta_10_2o * self.dt      # 第一关节角速度
        self.theta_21_1o += self.theta_21_2o * self.dt      # 第二关节相对角速度
        self.theta_20_1o = self.theta_10_1o + self.theta_21_1o  # 第二关节角速度
        self.X_0_1o += self.X_0_2o * self.dt                # 前进速度
        self.U = self.X_0_1o
        self.theta_10_1o_t.append(self.theta_10_1o)     # 记录
        self.theta_21_1o_t.append(self.theta_21_1o)
        self.theta_20_1o_t.append(self.theta_20_1o)
        self.X_0_1o_t.append(self.X_0_1o)

        # 两关节角度、前进位移
        self.theta_10 += self.theta_10_1o * self.dt
        self.theta_21 += self.theta_21_1o * self.dt
        self.theta_10 = np.clip(self.theta_10, -self.theta_10f, self.theta_10f)  # numpy.clip(a, a_min, a_max, out=None)[source],
        self.theta_21 = np.clip(self.theta_21, -self.theta_21f, self.theta_21f)  # clip这个函数将将数组中的元素限制在a_min, a_max之间
        self.theta_20 = self.theta_10 + self.theta_21
        self.X_0 += self.X_0_1o * self.dt
        self.theta_10_t.append(self.theta_10)   # 记录
        self.theta_21_t.append(self.theta_21)

        # 有用推进功率
        self.P_useful = (self.F_1x + self.F_2x) * self.U

        # 每隔一段时间积分计算有用功
        # if (self.counter % 100) == 99:
        #     self.W_useful = 0
        #     self.W_total = 0
        #     self.total_eta = 0
        self.W_total = self.W_total + (self.P_1 + self.P_2) * self.dt
        self.W_useful = self.W_useful + self.P_useful * self.dt

        # 计算效率
        if self.W_total == 0:
            self.total_eta = 0
        else:
            self.total_eta = self.W_useful / self.W_total

        ####################  数据保存（不要重复保存）  ####################
        # my_file = open('DDPG_state.txt', 'a')  # 保存 MAX_EP_STEPS = 200时的各运动变量数据
        # text = 'Speed: %f, angle1: %f, angle2: %f, v1: %f, v2: %f, M1: %f, M2: %f\n' % (
        #     self.X_0_1o, self.theta_10, self.theta_21, self.theta_10_1o, self.theta_21_1o, self.M_1, self.M_2)
        # my_file.write(text)
        # my_file.close()

        ####################  RL相关变量选取  ####################
        if(((self.P_1 + self.P_2) * self.dt) == 0):
            r = 0
        else:
            r = (self.P_useful * self.dt)/((self.P_1 + self.P_2) * self.dt)      # 顺时效率作为的奖励值
            # r = math.tanh(t)
        # self.fish_info['r'] = [self.theta_10_g-self.theta_10, self.theta_21_g-self.theta_21,self.theta_10_1o_g-self.theta_10_1o, self.theta_21_1o_g-self.theta_21_1o, self.theta_10_2o_g-self.theta_10_2o, self.theta_21_2o_g-self.theta_21_2o]
        self.fish_info['r'] = [self.theta_10, self.theta_21, self.theta_10_1o, self.theta_21_1o, self.theta_10_2o, self.theta_21_2o]
        s = self.fish_info['r'] # 运动变量作为状态

        if self.total_eta > 0.5:
            self.goal += 1
            if self.goal > 20:
                self.on_goal += 1
                done = False
                self.goal = 0
        else:
            self.goal = 0
        return s, r, done, self.M_1, self.M_2

        ####################  绘制图像  ####################
        # if (self.counter % 5000) == 0:
            # plt.figure(1)  # 绘制推进速度图
            # fig = plt.figure(1, figsize=(15, 10))
            # ax = fig.add_subplot(111)
            # ax.plot(self.X_0_1o_t, linewidth=2, label='Travelling Speed')  # 机器鱼前进速度
            # x = range(0, 5500, 500)
            # plt.xticks(x, ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'))
            # plt.tick_params(labelsize=40)
            # labels = ax.get_xticklabels() + ax.get_yticklabels()
            # [label.set_fontname('Times New Roman') for label in labels]
            # plt.ylabel('speed(m/s)', font)
            # plt.xlabel('time(s)', font)
            # plt.legend(loc='best', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
            # plt.savefig('./Figure1v_speed.pdf')
            # plt.savefig('./Figure1v_speed.png')
            #
            # plt.figure(2)   # 绘制关节角度图
            # plt.subplot(2, 1, 1)
            # plt.title('Angle of the fish')
            # plt.plot(self.theta_10_t, label='1st joint')  # 第一关节角度
            # plt.legend()
            # x = range(0, 5500, 500)
            # plt.xticks(x, ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'))
            # # plt.ylim(-0.5, 0.5)
            # plt.xlim(left=-100)
            # plt.ylabel('rad')
            #
            # plt.subplot(2, 1, 2)
            # plt.plot(self.theta_21_t,label='2nd joint',color='orange')    # 第二关节相对角度
            # x = range(0, 5500, 500)
            # plt.xticks(x, ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'))
            # plt.legend()
            # plt.ylim(-0.5, 0.5)
            # plt.xlim(left=-100)
            # plt.xlabel('t/s')

            # plt.figure(3)   # 绘制关节角速度图
            # plt.subplot(2, 1, 1)
            # plt.title('Angle Velocity of the fish')
            # plt.plot(self.theta_10_1o_t, label='1st joint')  # 第一关节角速度
            # plt.legend()
            # x = range(0, 5500, 500)
            # plt.xticks(x, ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'))
            # plt.ylim(-3, 3)
            # plt.xlim(left=-100)
            # plt.ylabel('rad/s')
            #
            # plt.subplot(2, 1, 2)
            # plt.plot(self.theta_21_1o_t, label='2nd joint', color='orange')
            # x = range(0, 5500, 500)
            # plt.xticks(x, ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'))
            # plt.legend()
            # plt.ylim(-4, 4)
            # plt.xlim(left=-100)
            # plt.xlabel('t/s')

            # plt.figure(4)   # 绘制关节角加速度图
            # plt.subplot(2, 1, 1)
            # plt.title('Angle Acceleration of the fish')
            # plt.plot(self.theta_10_2o_t,label='1st joint')
            # x = range(0, 5500, 500)
            # plt.xticks(x, ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'))
            # plt.legend()
            # # plt.ylim(-0.3, 0.3)
            # plt.xlim(left=-100)
            # plt.ylabel('rad/s^2')
            #
            # plt.subplot(2, 1, 2)
            # plt.plot(self.theta_21_2o_t,label='2nd joint',color='orange')
            # x = range(0, 5500, 500)
            # plt.xticks(x, ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'))
            # plt.legend()
            # # plt.ylim(-0.6, 0.6)
            # plt.xlim(left=-100)
            # plt.xlabel('t/s')

            # plt.figure(5)   # 绘制关节转矩图
            # plt.subplot(2, 1, 1)
            # plt.title('Torque of the fish')
            # plt.plot(self.M_1_t, label='1st joint')
            # plt.legend()
            # x = range(0, 5500, 500)
            # plt.xticks(x, ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'))
            # plt.ylim(-3, 3)
            # plt.xlim(left=-100)
            #
            # plt.subplot(2, 1, 2)
            # plt.plot(self.M_2_t, label='2nd joint', color='orange')
            # plt.legend()
            # x = range(0, 5500, 500)
            # plt.xticks(x, ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'))
            # plt.ylim(-0.8, 1)
            # plt.xlim(left=-100)
            # plt.xlabel('t/s')

            # plt.figure(6)   # 比例系数Kp
            # plt.subplot(2, 1, 1)
            # plt.plot(self.action0, label='Kp1')
            # plt.legend()
            # x = range(0, 5500, 500)
            # plt.xticks(x, ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'))
            # # plt.ylim(-3, 3)
            # plt.xlim(left=-100)
            #
            # plt.subplot(2, 1, 2)
            # plt.plot(self.action1, label='Kp2', color='orange')
            # plt.legend()
            # x = range(0, 5500, 500)
            # plt.xticks(x, ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'))
            # # plt.ylim(-0.4, 0.4)
            # plt.xlim(left=-100)
            # plt.xlabel('t/s')

            # plt.figure(7)
            # plt.subplot(1, 1, 1)
            # plt.plot(self.total_yita)
            # plt.show()
        # done and reward
        # print(self.X_0_1o)


    ####################  环境的重置与刷新  ####################
    def reset(self):    # 重置
        self.fish_info['r'] = np.zeros(6)
        self.theta_10 = 0
        self.theta_21 = 0 
        self.theta_10_1o = 0 
        self.theta_21_1o = 0 
        self.theta_10_2o = 0 
        self.theta_21_2o = 0 
        self.counter=0
        self.W_total = 0                                    
        self.W_useful = 0                                   
        self.total_eta = 0                                  
        self.reward = 0  
        return self.fish_info['r']
    def render(self):   # 刷新
        if self.viewer is None:
            self.viewer = Viewer(self.fish_info)
        self.viewer.render()
    # def sample_action(self):
    #     var = 1
    #     self.a[0] = np.clip(np.random.normal(100, var), 80, 100)  # 随机产生动作
    #     self.a[1] = np.clip(np.random.normal(20, var), 10, 30)
    #     return self.a  # two radians


####################  机器鱼仿真可视化（无需看亦无需修改）  ####################
class Viewer(pyglet.window.Window):
    bar_thc = 5
    def __init__(self, fish_info):
        # vsync = False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Fish', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.fish_info = fish_info
        self.center_coord = np.array([10, 200])

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.fish1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                  # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))        # color
        self.fish2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,
                     100, 160,
                     150, 160,
                     150, 150]),
            ('c3B', (249, 86, 86) * 4,))
        self.fish3 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [300, 150,
                     300, 160,
                     350, 160,
                     350, 150]),
            ('c3B', (249, 86, 86) * 4,))

    def render(self):
        self._update_fish()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_fish(self):
        a1l = a2l = a3l = 100
        (a2r, a3r) = self.fish_info['r'][0], self.fish_info['r'][1] # radian, angle
        a1xy = self.center_coord
        a1xy_ = self.center_coord + [50, 0]
        a2xy = a1xy_  # a2 start (x0, y0)
        a2xy_ = np.array([np.cos(a2r), np.sin(a2r)]) * a2l + a2xy   # a2 end and a3 start (x1, y1)
        a3xy_ = np.array([np.cos(a2r + a3r), np.sin(a2r + a3r)]) * a3l + a2xy_  # a3 end (x2, y2)

        a2tr, a3tr = np.pi / 2 - self.fish_info['r'][0], np.pi / 2 - self.fish_info['r'].sum()

        xy00 = a1xy - [0, self.bar_thc]
        xy01 = a1xy + [0, self.bar_thc]
        xy10 = a1xy_ + [0, self.bar_thc]
        xy11 = a1xy_ - [0, self.bar_thc]

        xy10_ = a2xy + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy11_ = a2xy + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy20 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc

        xy20_ = a2xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy21_ = a2xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy30 = a3xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy31 = a3xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc

        self.fish1.vertices = np.concatenate((xy00, xy01, xy10, xy11))
        self.fish2.vertices = np.concatenate((xy10_, xy11_, xy20, xy21))
        self.fish3.vertices = np.concatenate((xy20_, xy21_, xy30, xy31))

# if __name__ == '__main__':
#     env = FishEnv()
#     while True:
#         s = env.reset()
#         i = 0
#         for i in range(4000):
#             env.render()
#             env.step(env.sample_action())
