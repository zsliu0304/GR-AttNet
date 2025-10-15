# test.py
import os
import time
import logging
from datetime import datetime

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.filters import gaussian

from network import GRAttNet
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.dataset_processing.grasp import detect_grasps

# ===================== Parameters =====================
ROOT_DATASET = 'D:/cornell'                # dataset root
PTH_PATH     = './checkpoints/best_iou.pth'  # trained weights
SAVE_DIR     = './test_results'            # visualization folder
SPLIT_RATIO  = 0.01                         # last 1% for test
N_GRASPS     = 2                            # number of grasp rectangles to draw
IOU_THR      = 0.25
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(os.path.join(SAVE_DIR, 'test.log'), mode='w'),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ------------------ Post-processing ------------------
def post_process_output(q_img, cos_img, sin_img, width_img):
    q_img     = q_img.cpu().numpy().squeeze()
    ang_img   = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    q_img     = gaussian(q_img,     2.0, preserve_range=True)
    ang_img   = gaussian(ang_img,   2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)
    return q_img, ang_img, width_img


# ------------------ Visualization ------------------
def plot_results(fig, rgb_img, grasp_q_img, grasp_angle_img,
                 att_img=None, no_grasps=1, grasp_width_img=None,
                 save_path=None):
    """2x2 grid: RGB | Attention, Grasp | Q"""
    gs = detect_grasps(grasp_q_img, grasp_angle_img,
                       width_img=grasp_width_img, no_grasps=no_grasps)

    plt.clf()
    # row 1
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)          # *** attention map ***
    if att_img is not None:
        if isinstance(att_img, torch.Tensor):
            att_img = torch.sigmoid(att_img).cpu().numpy().squeeze()
        att_img = cv2.resize(att_img, (rgb_img.shape[1], rgb_img.shape[0]))
        ax.imshow(att_img, cmap='jet', vmin=0, vmax=1)
        ax.set_title('Attention')
    else:
        ax.axis('off')
    ax.axis('off')

    # row 2
    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax, color='r')
    ax.set_title('Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot, ax=ax, fraction=0.046)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f'Result saved -> {save_path}')


# ------------------ Main test ------------------
def main():
    # 1. dataset
    Dataset = get_dataset("cornell")
    test_dataset = Dataset(ROOT_DATASET,
                           output_size=224,
                           ds_rotate=True,
                           random_rotate=False,
                           random_zoom=False,
                           include_depth=True,
                           include_rgb=True)

    indices     = list(range(test_dataset.length))
    split       = int(np.floor(SPLIT_RATIO * test_dataset.length))
    val_indices = indices[split:]
    val_loader  = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices)
    )

    # 2. model (instantiate then load weights)
    net = GRAttNet().to(DEVICE)
    state_dict = torch.load(PTH_PATH, map_location=DEVICE)
    net.load_state_dict(state_dict)
    net.eval()
    logger.info(f'Weights loaded from {PTH_PATH}')

    # 3. evaluation & visualization
    results = {'correct': 0, 'failed': 0}
    start_time = time.time()

    pbar = tqdm(val_loader, total=len(val_loader), desc='Testing')
    for idx, (x, y, didx, rot, zoom) in enumerate(pbar):
        x, y = x.to(DEVICE), [yy.to(DEVICE) for yy in y]
        with torch.no_grad():
            loss_dict = net.compute_loss(x, y)

        q_img, ang_img, w_img = post_process_output(
            loss_dict['pred']['pos'],
            loss_dict['pred']['cos'],
            loss_dict['pred']['sin'],
            loss_dict['pred']['width'])

        # IoU evaluation
        is_match = evaluation.calculate_iou_match(
            q_img, ang_img,
            val_loader.dataset.get_gtbb(didx, rot, zoom),
            no_grasps=N_GRASPS,
            grasp_width=w_img,
            threshold=IOU_THR)
        results['correct' if is_match else 'failed'] += 1

        # attention map (fallback to Q map if None)
        att_map = loss_dict['pred'].get('att', None)
        if att_map is None:
            att_map = torch.tensor(q_img).unsqueeze(0).unsqueeze(0)

        # visualization
        fig = plt.figure(figsize=(8, 6))
        plot_results(fig,
                     rgb_img=test_dataset.get_rgb(didx, rot, zoom, normalise=False),
                     grasp_q_img=q_img,
                     grasp_angle_img=ang_img,
                     att_img=att_map,
                     no_grasps=N_GRASPS,
                     grasp_width_img=w_img,
                     save_path=os.path.join(SAVE_DIR, f'result_{idx:04d}.png'))
        plt.close(fig)

        pbar.set_postfix({'IoU': f"{results['correct']/(sum(results.values()) or 1):.3f}"})

    # 4. summary
    total   = results['correct'] + results['failed']
    avg_iou = results['correct'] / total if total else 0.0
    avg_ms  = (time.time() - start_time) / total * 1000
    logger.info(f'IoU Results: {results["correct"]}/{total} = {avg_iou:.4f}')
    logger.info(f'Average inference time per image: {avg_ms:.2f} ms')


if __name__ == '__main__':
    main()