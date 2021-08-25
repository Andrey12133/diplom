import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate
import math
import statistics
import matplotlib.gridspec as gridspec
import dispersity
import pickle
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          # 'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

from mpl_toolkits.mplot3d import Axes3D

# смотрит по одной фотографии концентрацию капель и монодисперсность



# выдает все нужную информацию из массива данных.
def general_inf(data_new):
    vel_x_nn = {}
    y_nn = {}
    vel_x_y = {}
    y_y = {}
    y_n = []
    vel_x_n = []
    y = {}
    yyyy = {}
    for it in data_new:
        vel_x_nn[it] = []
        y_nn[it] = []
        vel_x_y[it] = []
        y_y[it] = []
        y[it] = []
        yyyy[it] = []
        if len(data_new[it]) > 60:
            for i in range(2, len(data_new[it])):
                vel_x_nn[it].append(math.sqrt((data_new[it][i][0]-data_new[it][i-1][0])**2+(data_new[it][i][1]-data_new[it][i-1][1])**2))
                y_nn[it].append(data_new[it][i][1])
                y[it].append((data_new[it][i][1]-data_new[it][0][1])**2)
                yyyy[it].append(data_new[it][i][1]-data_new[it][0][1])
            a = np.mean(vel_x_nn[it])
            b = statistics.stdev(vel_x_nn[it])
            # b = a/6
            n = 2.5
            # for j, k in enumerate(vel_x_nn[it]):
            #     if (k <= a + n*b) and (k >= a - n*b):
            #         vel_x_y[it].append(k)
            #         y_y[it].append(y_nn[it][j])

            y_n.append(np.mean(y_nn[it])*0.91/0.5)
            #[y] = mkm
            #[velx_n] = mm/c
            vel_x_n.append(np.mean(vel_x_nn[it])*0.91/0.5*99.4/1000)


    # print(y_n, vel_x_n, y)
    return y_n, vel_x_n, y


def show_gen_inf(y_n, vel_x_n, fig, k=0):

    ax1 = fig.add_subplot(spec2[k, 0])
    ax1.set(xlabel='Beans, высота y', ylabel='Probability')
    ax1.hist(y_n, bins=30)

    ax2 = fig.add_subplot(spec2[k, 1])
    ax2.set(xlabel='Beans, скорось v', ylabel='Probability')
    ax2.hist(vel_x_n, bins=30)

    ax3 = fig.add_subplot(spec2[k, 2])
    for i in range(0, len(y_n)):
        ax3.plot(y_n[i], vel_x_n[i], '+')
    ax3.set(xlabel='y, μm', ylabel='velosity, μm/s')
    # fig.suptitle('___', fontsize=18)
    # fig.subplots_adjust(wspace=0.2)


def deffusion(ym, mk, axs_3, v):
    # fig_3 = plt.figure()
    # axs_3 = fig_3.add_subplot(111)
    duffusion_line = []
    for j in range(mk):
        hh = 0
        for i in ym:
            hh += ym[i][j]
        hh /= len(ym)
        duffusion_line.append(hh)
    print(len(duffusion_line))
    line = axs_3.plot([gg*1/99.4 for gg in range(len(duffusion_line))], duffusion_line, '-', label=f'{legens[v]}', linewidth=2)
    # axs_3.set_title(f'{name_pic_2}')
    axs_3.set_xlabel('t, с')
    axs_3.set_ylabel('y, мм^2')
    # axs_3.set_xlim([0, len(duffusion_line)*1/99.4])
    axs_3.set_ylim([1, max(duffusion_line)])
    axs_3.set_yscale('log')
    axs_3.set_xscale('log')
    axs_3.grid()

    axs_3.legend()

    # fig_3.savefig(r'2021.04.16_acryl_cont_deff_3' + '.png', dpi=100)
    # fig_3.subplots_adjust(wspace=0.1, hspace=0.2)

    # for i in ym:
    #     line = axs_3.plot([gg for gg in range(len(ym[i]))], ym[i], '-')


def show_needed_inf(y_n, vel_x_n, s, cony, axs, kkkhh):
    global itera, li, name, bb

    l_d = 1536
    y, x, y_f, x_f = [], [], [], []
    for i in range(s):
        # print(i)
        gg = False
        for j, t in enumerate(y_n):
            # print(min(y_n))
            # print(t, j, vel_x_n[j])
            if (t >= l_d * i / s) and (t < l_d * (i+1) / s):
                y.append(t)
                x.append(vel_x_n[j])
                gg = True
        if x:
            a = np.mean(x)
            b = 0.2 * a
            v = 0
            for f, h in enumerate(x):
                if a - v*b <= h <= a + v*b:
                    print('work', h, a)
                    x.pop(f)
                    y.pop(f)
            # cony.append(len(y))
        if gg and x:
            x_f.append(np.mean(x))
            y_f.append(np.mean(y))
        # else:
            # x_f.append(0)
            # y_f.append(l_d / s * (i+0.5))
        #         # /1.68
        y.clear()
        x.clear()
    itera += 1
    clolour = ['red', 'blue']
    namesss = ['эмульсия', 'суспензия']
    # vel_all_chanel = False
    if vel_all_chanel:
        nx = np.linspace(1, 34, 11)
        xnew = np.linspace(min(y_f), max(y_f), 500)
        # z = nx[kkkhh]
        # print(z)
        z = np.full((len(y_f), len(x_f)), nx[kkkhh])
        # print(z)
        # # print(y_f)
        # # print(x_f)
        f1 = interpolate.interp1d(y_f, x_f, kind='cubic')
        # dot = axs.scatter(y_f, x_f, marker='+', color=f'{clolour[itera-1]}')
        # dot2 = axs.plot(xnew, f1(xnew), '-', color=f'{clolour[itera-1]}', label=f'{namesss[itera-1]}', linewidth=0.8)
        dot2 = axs.plot(xnew, f1(xnew), '-', linewidth=1.5, label=f'{itera}')
        # dot2 = axs.plot(y_f, x_f, '-', linewidth=1.5, label=f'{itera}')
        # dot2 = axs.plot(xnew, f1(xnew), 'b-', linewidth=1.5)
        # label = f'{kkkhh}'
        # axs.contour(y_f, x_f, z)
        # axs.plot_wireframe(y_f, x_f, z, linewidth=0.8)
        # axs.plot_surface(y_f, x_f, z)
        # dot22 = axs.plot(y_f[x_f.index(max(x_f))], max(x_f), 'o', color='red')
        # dot33 = axs.plot(y_f[s//2-1], x_f[s//2-1], 'o', color='blue')
        axs.set_ylabel('V, мм/с')
        axs.set_xlabel('Y-координата, мкм')
        # axs.set_zlabel('X-coordinate, mm')
        if max(x_f) != 0:
            li.append(x_f[s//2]/max(x_f))
        else:
            li.append(1)

    if concent:
        axee = fig5.add_subplot(spec2[0, bb])

        twin1 = axee.twinx()

        # cony, nx = dispersity.a(s, bb)
        # print(len(cony), len(y_f))
        # nx = np.linspace(0, 800, s*4)
        # f3 = interpolate.interp1d(y_f, cony, kind='cubic')
        f4 = interpolate.interp1d(y_f, x_f, kind='linear')
        yyy = np.linspace(64, 706, 12)
        # axs_2.plot(nx2, f3(nx2), '--')
        # print(min(y_n), max(y_n))
        # print(len(y_f), len(cony), len(y_n))
        # yn = f3(nx)
        # print(len(nx), len(yn))
        # p2 = twin1.plot(nx, f4(nx), label='V, мм/с', lw=0.8, color="purple", ls='-', marker='.')
        p2 = twin1.plot(y_f, x_f, '-y', label='Velocity', linewidth=1)

        p1, = axee.plot(yyy, cony, '--g', label='Density', linewidth=0.8)
        # p2, = twin1.plot(y_f, x_f, label='V, мм/с', linewidth=0.8, color="red", ls='-', marker='.')

        p3, = axee.plot(yyy, cony, '*g')
        p4, = twin1.plot(y_f, x_f, '*r')

        bb += 1

        axee.set_ylabel("Плотность, 1/мкм")
        axee.set_xlabel("Y-координата, мкм")
        twin1.set_ylabel("V, мм/сек")

        # axee.legend(handles=[p1, p2])

        # fig5.add_subplot(spec2[0, bb]).plot(y_f, cony, '*')
        # fig5.add_subplot(spec2[0, bb]).plot(y_f, lle, 'o')


def chch(ggg):
    bbb = []
    ww = 0
    for qq in range(12):
        for e in ggg:
            ww += ggg[e][qq]
        ww /= len(ggg)
        bbb.append(ww)
    return bbb

# графики по отношеню скоростей
# DIR = r'C:\Users\mayda\PycharmProjects\OpenCV\mytest\2021.04.19\acryl\nocontrol_oridge'
# DIR = r'C:\Users\mayda\PycharmProjects\OpenCV\mytest\2021.04.19\acryl\control_oridge'
# DIR = r'C:\Users\mayda\PycharmProjects\OpenCV\mytest\2021.04.19\acryl\nocontrol_oridge_heat'
DIR = r'D:\Microfluidics\Data\Velosity_drops_profiles\21.06.23\ends'
# DIR2 = r'C:\Users\mayda\PycharmProjects\OpenCV\mytest\diplom\21.04.19\control_smal_heat_conc'
plus_name = 'ends'
# name_pic_2 = '2021.04.19_acryl_nocontrol_oridge_' + plus_name
# name_pic_2 = '2021.04.19_new_acryl_control_oridge_'+ plus_name
# name_pic_2 = '2021.04.19_new_acryl_nocontrol_oridge_hea_' + plus_name
name_pic_2 = 'AA_oil' + plus_name
# name_pic_2 = 'fff'
itera = 0
s = 30
dot_1 = 0
li = []
bluzdania = False
vel_all_chanel = True
# концентрация и распределение по скоростям
# DIR_4 = r'C:\Users\mayda\PycharmProjects\OpenCV\mytest\2021.03.05_heat_2'
DIR_4 = DIR
name_pic1 = name_pic_2 #+ 'conc'
scater_var = False
concent = False

# графики по диффузии
# количество кадров, для участии в
mm = 400
# path_toch = r'C:\Users\mayda\PycharmProjects\OpenCV\mytest\09.02.2021_1chip_control'
path_toch = DIR
name_pic = name_pic1 #+ 'deff'
deff_point_cheak = False
deff_multipal_cheak = False

if not os.path.isdir('figures\\summer_work\\'):
    os.mkdir('figures\\summer_work\\')

os.chdir('figures\\summer_work\\')

if concent:
    kkk = []
    for name in os.listdir(DIR+'\\'):
        path = os.path.join(DIR + '\\' + name)
        kkkhh = 0
        print(f'{name}')
        with open(path) as json_file:
            raspe = json.load(json_file)
            kkk.append(chch(raspe))

    bb = 0
    fig5 = plt.figure(constrained_layout=True, figsize=(13, 4))
    spec2 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig5)
    for name in os.listdir(DIR_4+'\\'):
        path = os.path.join(DIR_4 + '\\' + name)
        print(f'{name}')
        with open(path) as json_file:
            js = json.load(json_file)
            r1, r2, gy = general_inf(js)
        show_needed_inf(r1, r2, s, kkk[bb])

    fig5.savefig(name_pic1 + '.png', dpi=100)

if vel_all_chanel:
    kkkhh = 0
    fig_2 = plt.figure(figsize=(4, 5))
    # fig_2 = plt.figure()
    # axs_2 = fig_2.add_subplot(111)
    # axs = fig_2.add_subplot(111, projection='3d')
    axs = fig_2.add_subplot(111)

    for name in os.listdir(DIR+'\\'):
        path = os.path.join(DIR + '\\' + name)
        print(f'{name}')
        kkkhh += 1
        with open(path) as json_file:
            js = json.load(json_file)
            r1, r2, gy = general_inf(js)
        show_needed_inf(r1, r2, s, dot_1, axs, kkkhh)
        # show_gen_inf(r1, r2)

    # vel_all_chanel = False
    if vel_all_chanel:
        l = len(os.listdir(DIR))
        print(l)
        # nx = np.linspace(1, 34, l)
        # nx2 = np.linspace(1, 34, l*4)
        # f2 = interpolate.interp1d(nx, li, kind='cubic')
        # axs_2.plot(nx2, f2(nx2), '--', nx, li, '*', linewidth=0.8, color='black')
        # axs_2.set_xlabel('X-coordinate, mm')
        # axs_2.set_ylabel('Ratio Vm/V0')
        # axs_2.set_xlim([10, 23])
        # axs.plot(nx, li, '--')
        # axs.plot(nx, li, 'o')

    first_legend = plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig_2.tight_layout()
    # axs_2.grid()

    axs.grid()
    # axs.set_ylim([0.3, 0.6])
    # axs.set_xlim([0, 1050])

    # plt.title(f'{s} дроблений, отношение {dot_1}-й к точке в середине')
    # fig_2.subplots_adjust(wspace=0.9, hspace=0)

    fig_2.savefig(name_pic_2 + '.png', dpi=100)
    # fig_2.subplots_adjust(wspace=0.1, hspace=0.2)


if scater_var:
    fig4 = plt.figure(constrained_layout=True, figsize=(10, 7))
    spec2 = gridspec.GridSpec(ncols=3, nrows=3, figure=fig4)

    count = 0
    for name in os.listdir(DIR_4+'\\'):
        path = os.path.join(DIR_4 + '\\' + name)
        print(f'{name}')
        with open(path) as js_file:
            js = json.load(js_file)
            r1, r2, gy = general_inf(js)
        show_gen_inf(r1, r2, fig4, count)
        count += 1
    fig4.savefig(name_pic1 + '.png', dpi=100)


# точечная проверка

if deff_point_cheak:
    path_toch = r'C:\Users\mayda\PycharmProjects\OpenCV\mytest\2021.04.16_acryl_cont'
    with open(path_toch) as js_file:
        fig, axs = plt.subplots(figsize=(12, 3))
        # fig = plt.figure()
        # axs = fig.add_subplot(figsize=(15, 9))
        # jjs = pickle.load(js_file)
        jjs = json.load(js_file)
        e1, e2, ey = general_inf(jjs)
        k = {}
        for h in ey:
            if len(ey[h]) > mm:
                k[h] = ey[h]
        print(k)
        deffusion(k, mm, fig, axs)
    fig.savefig(name_pic + '.png', dpi=100)

# сшивка диффузий
if deff_multipal_cheak:
    legens = ['начало', 'центр', 'конец']
    i = 0
    fig, axs = plt.subplots()
    # fig = plt.figure()
    # axs = fig.add_subplot(111)
    for name in os.listdir(path_toch+'\\'):
        # with open(path_toch+'\\'+name, 'rb') as f:
        #     jjs = pickle.load(f)
        with open(path_toch+'\\'+name) as js_file:
            jjs = json.load(js_file)
            e1, e2, ey = general_inf(jjs)
            k = {}
            for h in ey:
                if len(ey[h]) > mm:
                    k[h] = ey[h]
            deffusion(k, mm, axs, i)
            i += 1

    # plt.tight_layout()
    fig.savefig(name_pic + '.png', dpi=100)

if bluzdania:
    fig__2 = plt.figure(figsize=(17, 3))
    axsss = fig__2.add_subplot(111)
    for name in os.listdir(path_toch+'\\'):
        with open(path_toch+'\\'+name) as js_file:
            jjs = json.load(js_file)
            e1, e2, ey, yyyy = general_inf(jjs)
            # ht = str(9107)
            # print(ey)
            for k in yyyy:
                if len(yyyy[k]) > 363:
                    xx = np.linspace(0, len(yyyy[k])/99, len(yyyy[k]))
                    axsss.plot(xx, yyyy[k])
                    axsss.grid()
                    # axsss.set_ylim([0, 0.7])
                    # axsss.set_xlim([0, 1050])




            # xx = np.linspace(0, len(yyyy[ht])/99, len(yyyy[ht]))
            # axsss.plot(xx, yyyy[ht])
plt.show()
