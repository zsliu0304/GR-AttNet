import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from .grasp import GraspRectangles, detect_grasps


def plot_output(fig, rgb_img, grasp_q_img, grasp_angle_img, attention_output, no_grasps=1, grasp_width_img=None):
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()

    # RGB Image with Grasps
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    # Attention Output
    ax = fig.add_subplot(2, 2, 2)
    plot = ax.imshow(attention_output, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Attention')
    ax.axis('off')
    plt.colorbar(plot)

    # Q Output
    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    # Angle Output
    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)

    plt.pause(0.1)
    fig.canvas.draw()


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, threshold=0.25):
    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs

    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > threshold:
            return True
    return False