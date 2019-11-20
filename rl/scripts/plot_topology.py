import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pycigar.config as config
import os
from datetime import datetime


def draw_circle(x, y, r, portion, rl=False):
    if rl:
        color = 'b'
    else:
        color = 'g'
    theta = np.linspace(0, 2*np.pi, 100)
    p1 = r*np.cos(theta) + x
    p2 = r*np.sin(theta) + y
    for i in range(len(theta)):
        plt.plot([x, p1[i]], [y, p2[i]], color)

    for i in range(74, 74+portion):
        if i < 100:
            plt.plot([x, p1[i]], [y, p2[i]], 'r')
        else:
            plt.plot([x, p1[i-100]], [y, p2[i-100]], 'r')


def plot_topology(hack):
    topo = os.path.join(config.PROJECT_PATH, 'scripts/Topology-of-the-IEEE-34-bus-test-feeder.png')
    img = plt.imread(topo)
    f = plt.figure(figsize=(10, 10))
    plt.imshow(img)
    #draw_circle(45, 153, 7, hack)  # 802
    #draw_circle(82, 153, 7, hack)  # 806
    #draw_circle(120, 153, 7, hack)  # 808
    #draw_circle(120, 220, 7, hack)  # 810
    #draw_circle(158, 153, 7, hack)  # 812
    #draw_circle(195, 153, 7, hack)  # 814
    #draw_circle(275, 153, 7, hack)  # 850
    #draw_circle(316, 153, 7, hack)  # 816
    #draw_circle(316, 112, 7, hack)  # 818
    #draw_circle(316, 75, 7, hack)  # 820
    draw_circle(316, 57, 7, hack)  # 822m
    draw_circle(316, 40, 7, hack)  # 822
    draw_circle(353, 153, 7, hack)  # 824
    draw_circle(372, 153, 7, hack, True)  # 826m
    draw_circle(389, 153, 7, hack, True)  # 826
    draw_circle(353, 286, 7, hack)  # 828
    draw_circle(383, 286, 7, hack)  # 830m
    draw_circle(408, 286, 7, hack)  # 830
    #draw_circle(446, 286, 7, hack)  # 854
    #draw_circle(495, 286, 7, hack)  # 856
    #draw_circle(446, 249, 7, hack)  # 852
    #draw_circle(446, 190, 7, hack)  # 832
    #draw_circle(525, 190, 7, hack)  # 888
    #draw_circle(565, 190, 7, hack)  # 890
    #draw_circle(446, 153, 7, hack)  # 858
    #draw_circle(446, 113, 7, hack)  # 864
    #draw_circle(492, 153, 7, hack)  # 834
    #draw_circle(492, 112, 7, hack)  # 842
    #draw_circle(492, 76, 7, hack)  # 844
    #draw_circle(492, 39, 7, hack)  # 846
    #draw_circle(492, 5, 7, hack)  # 848
    draw_circle(492, 153, 7, hack)  # 834
    #draw_circle(565, 153, 7, hack)  # 860
    draw_circle(623, 153, 7, hack)  # 836
    #draw_circle(623, 190, 7, hack)  # 862
    draw_circle(623, 230, 7, hack)  # 838
    draw_circle(623, 210, 7, hack)  # 838m
    draw_circle(660, 153, 7, hack)  # 840

    draw_circle(180, 60, 30, hack)  # 822 description
    plt.plot([182, 310], [29, 33], 'k--', linewidth=1)
    plt.plot([192, 310], [89, 49], 'k--', linewidth=1)
    if hack == 0:
        plt.text(130, 26, '          Pre-attack\nNo DER compromised', fontsize=8)
    else:
        plt.text(130, 26, '          Attack\n{}% of all DER compromised'.format(hack), fontsize=8)

    save_path = os.path.join(config.LOG_DIR, 'topology_{}.png'.format(datetime.now().strftime("%H:%M:%S.%f_%d-%m-%Y")))
    f.savefig(save_path)
    plt.close(f)
