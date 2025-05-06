"""
Build the basic framework for main.py, DDPG.py and env.py.
"""
import matplotlib
from env import FishEnv
from ddpg import DDPG
import matplotlib.pyplot as plt
import numpy as np
import time

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

####################    ####################
MAX_EPISODES = 100 #原来100
MAX_EP_STEPS = 100
EVAL_EPISODE = 10000
KP_1 = 20
KP_2 = 50

ON_TRAIN = True
# ON_TRAIN = False

ON_DDPG = True
# ON_DDPG = False

env = FishEnv()     # set env
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

rl = DDPG(a_dim, s_dim, a_bound)        # set RL method
eta = []
counter = []


####################    ####################
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0
        for j in range(MAX_EP_STEPS):
            env.render()

            a = rl.choose_action(s)

            s_, r, done, M1, M2 = env.step(a)

            rl.store_transition(s, a, r, s_)
            # print(rl.gotq(s, a))
            ep_r = r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()
            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.3f | steps: %i' % (i, '---' if not done else 'done', ep_r, j))
                # my_file = open('data.txt','a')
                # text = 'Ep: %i | %s | ep_r: %.1f | steps: %i\n' % (i, '---' if not done else 'done', ep_r, j)
                # my_file.write(text)
                # my_file.close()

                ####################  数据保存（不要重复保存）  ####################
                # 每次改变 MAX_EP_STEPS 后保存一次效率数据
                # # my_file = open('DDPG_eta_50.txt', 'a')      # MAX_EP_STEPS = 50 时的效率
                # # my_file = open('DDPG_eta_100.txt', 'a')     # MAX_EP_STEPS = 100 时的效率
                # # my_file = open('DDPG_eta_150.txt', 'a')     # MAX_EP_STEPS = 150 时的效率
                # my_file = open('DDPG_eta_200.txt', 'a')     # MAX_EP_STEPS = 200 时的效率
                # # my_file = open('DDPG_eta_250.txt', 'a')     # MAX_EP_STEPS = 250 时的效率
                # text = 'Ep: %i | ep_r: %.1f\n' % (i, ep_r)
                # my_file.write(text)
                # my_file.close()


                counter.append(i)
                if ep_r > 100 or ep_r < 0:
                    eta.append(0)
                else:
                    eta.append(ep_r)
                break


    ####################  绘制图像  ####################
    # plt.figure(1)
    # plt.subplot(1, 1, 1)
    # plt.plot(self.X_0_1o_t)
    # plt.title('Head Speed')
    # plt.ylabel('m/s')
    # plt.xlabel('episode')

    # plt.figure(2)
    # plt.subplot(2, 2, 1)
    # plt.plot(env.theta_10_t,label='1st Angular Velocity')
    # plt.legend()
    # plt.subplot(2, 2, 3)
    # plt.plot(env.theta_10_g_t,label='1st Angular Acceleration',color='orange')
    # plt.legend()
    # plt.xlabel('t/s')
    # plt.subplot(2, 2, 2)
    # plt.plot(self.theta_21_t,label='2st Angular Velocity')
    # plt.legend()
    # plt.subplot(2, 2, 4)
    # plt.plot(self.theta_21_g_t,label='2st Angular Acceleration',color='orange')
    # plt.legend()
    # plt.xlabel('episode')


    fig = plt.figure(8, figsize=(15, 10))
    ax = fig.add_subplot(111)
    # ax.set_xticks(['0', '5', '10', '15', '20', '25', '30'])
    # plt.ylim(-3, 60)
    # plt.xlim(-100, 3100)
    ax.plot(eta, linewidth=2, label='Propulsion Efficiency')
    plt.subplots_adjust(left=0.12, right=0.9, top=0.9, bottom=0.13)
    ax.set_xlabel('Episode', font)
    ax.set_ylabel('efficiency (%)', font)
    plt.tick_params(labelsize=40)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
    # plt.savefig('./Fig01_DDPG_eta.pdf')
    # plt.savefig('./Fig01_DDPG_eta.png')


    # plt.figure(4)
    # plt.subplot(2, 1, 1)
    # plt.plot(self.action0)
    # plt.subplot(2, 1, 2)
    # plt.plot(self.action1)
    plt.show()


    rl.save()


####################  重置与刷新  ####################
def eval():
    total_eta_t = []                            # 推进效率
    kp1_array = []
    kp2_array = []
    rl.restore()

    env.render()
    env.viewer.set_vsync(True)
    
    # while True:
    s = env.reset()
    ep_r = 0
    cnt_eval = 0
    for _ in range(EVAL_EPISODE):
        env.render()
        cnt_eval += 1
        a = rl.choose_action1(s) if ON_DDPG else [KP_1, KP_2]
        
        kp1_array.append(a[0])
        kp2_array.append(a[1])

        s, r, done, m1, m2 = env.step(a)
        ep_r=r

        total_eta_t.append(100*ep_r)

        # if done:
        #     break
        print(f'[EVAL {cnt_eval/EVAL_EPISODE*100:.2f}%] eta: {ep_r:.4f}')
        
    # 绘制Kp图
    fig = plt.figure(2, figsize=(15, 10))
    ax1 = fig.add_subplot(211)
    x = range(0, 500, 100)
    # plt.xticks(x, ('0','5','10'))
    plt.ylim(0, 25)
    plt.xlim(-50, 550)
    ax1.plot(kp1_array, linewidth=2, label='1st joint')
    ax1.set_ylabel('angle(rad)',font)
    plt.tick_params(labelsize=40)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)

    ax2 = fig.add_subplot(212)
    x = range(0, 500, 100)
    # plt.xticks(x, ('0','5','10'))
    plt.ylim(35, 65)
    plt.xlim(-50, 550)
    ax2.plot(kp2_array, linewidth=2, label='2nd joint', color='orange')
    ax2.set_xlabel('time(s)',font)
    plt.tick_params(labelsize=40)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(loc='lower right', prop=font_1, labelspacing=0.1, handletextpad=0.1, borderpad=0.2, handlelength=1.0)
    # plt.savefig('./Figure2_angle.pdf')
    # plt.savefig('./Figure2_angle.png')
    plt.show()

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

    # 绘制关节角度图
    # 创建一个 Figure 和单个 Axes
    fig, ax = plt.subplots(figsize=(15, 10))

    # 设置 x 轴刻度
    x = range(0, 500, 100)
    ax.set_xticks(x)
    # ax.set_xticklabels(('0', '5', '10', '15', '20', '25', '30'))

    # 限定坐标轴范围
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlim(-50, 550)

    # 绘图
    ax.plot(env.theta_10_t, linewidth=2, label='1st joint')
    ax.plot(env.theta_21_t, linewidth=2, label='2nd joint', color='orange')

    # 设置标签字体
    ax.set_ylabel('angle(rad)', fontdict=font)

    # 调整刻度字体大小和字体
    ax.tick_params(labelsize=40)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # 添加图例
    ax.legend(loc='lower right',
            prop=font_1,
            labelspacing=0.1,
            handletextpad=0.1,
            borderpad=0.2,
            handlelength=1.0)

    plt.show()
    

if ON_TRAIN:
    train()
    # p=env.reset()
    # start_time = time.time()
    # b = rl.choose_action(p)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"执行时间: {execution_time} 秒")
    # print(b)
else:
    eval()
