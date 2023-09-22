import matplotlib.pyplot as plt

def plot_two_subfig(rgb_data_plot, rgb_ideal_plot, rgb_apply_ccm_plot):
    """[all input should be in shape[3, num-of-points].]
    Args:
        rgb_data_plot ([type]): [original rgb data]
        rgb_ideal_plot ([type]): [real rgb (should be linear)]
        rgb_apply_ccm_plot ([type]): [corrected rgb data by CCM]
    """
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection='3d')
    # fig3 = plt.figure(3)
    # ax3 = fig3.add_subplot(111, projection='3d')


    #------------1 fig-------------------------------------
    # rgb_data_plot = rgb_data.permute(1,0)
    x2 = rgb_data_plot[0]
    y2 = rgb_data_plot[1]
    z2 = rgb_data_plot[2]

    ax1.scatter(x2, y2, z2, marker='*', c='b', label='origin RGB')

    # ax1.set_xlim(-80, 360)
    # ax1.set_ylim(-80, 360)
    # ax1.set_zlim(-80, 360)
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax1.set_zlim(0,1)
    ax1.set_xlabel('R')
    ax1.set_ylabel('G')
    ax1.set_zlabel('B')

    # rgb_ideal_plot = rgb_ideal.permute(1, 0)
    x3 = rgb_ideal_plot[0]
    y3 = rgb_ideal_plot[1]
    z3 = rgb_ideal_plot[2]

    ax1.scatter(x3, y3, z3, marker='o', c='c', label='target rgb')

    for i in range(len(x3)):
        ax1.plot([x2[i], x3[i]], [y2[i], y3[i]], [z2[i], z3[i]], 'k-.')
    ax1.legend()



    #-------------2 fig ----------------------------------------------
    # ax2.set_xlim(-80, 360)
    # ax2.set_ylim(-80, 360)
    # ax2.set_zlim(-80, 360)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.set_zlim(0,1)
    ax2.set_xlabel('R')
    ax2.set_ylabel('G')
    ax2.set_zlabel('B')
    ax2.scatter(x3, y3, z3, marker='o', c='c', label='target rgb')

    # rgb_apply_ccm_plot = rgb_apply_ccm.permute(1,0)
    x4 = rgb_apply_ccm_plot[0]
    y4 = rgb_apply_ccm_plot[1]
    z4 = rgb_apply_ccm_plot[2]
    ax2.scatter(x4, y4, z4, marker='^', c='r', label='apply ccm rgb')

    for i in range(len(x3)):
        ax2.plot([x4[i], x3[i]], [y4[i], y3[i]], [z4[i], z3[i]], 'k-.')

    ax2.legend()


    #-------------------3 fig-------------------------------------------------
    # ax3.set_xlim(0,1)
    # ax3.set_ylim(0,1)
    # ax3.set_zlim(0,1)
    # ax3.set_xlabel('R')
    # ax3.set_ylabel('G')
    # ax3.set_zlabel('B')
    # ax3.scatter(x2, y2, z2, marker='*', c='b', label='origin RGB')
    # ax3.scatter(x4, y4, z4, marker='^', c='yellow', label='apply ccm rgb')

    # for i in range(len(x2)):
    #     ax3.plot([x2[i], x4[i]], [y2[i], y4[i]], [z2[i], z4[i]], 'k-.')
    # ax3.legend()


    plt.show()