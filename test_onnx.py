# test_onnx.py
import os
import time
import logging
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.filters import gaussian
import onnxruntime as ort

from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.dataset_processing.grasp import detect_grasps

# ================= Parameters =================
ROOT_DATASET = 'D:/cornell'
ONNX_PATH    = './grattnet.onnx'
SAVE_DIR     = './onnx_results'
OUTPUT_SIZE  = 224
BATCH_SIZE   = 1
N_GRASPS     = 2
IOU_THR      = 0.25
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(SAVE_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(os.path.join(SAVE_DIR, 'onnx_test.log'), mode='w'),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ---------------- Post-processing (squeeze batch dim only) ----------------
def post_process_output(q_img, cos_img, sin_img, width_img):
    # remove batch dim, keep spatial dims
    q_img     = q_img.squeeze(0)         # (1,H,W) -> (H,W)
    cos_img   = cos_img.squeeze(0)
    sin_img   = sin_img.squeeze(0)
    width_img = width_img.squeeze(0)     # (1,H,W) -> (H,W)

    ang_img   = np.arctan2(sin_img, cos_img) / 2.0
    width_img = width_img * 150.0

    q_img     = gaussian(q_img,     2.0, preserve_range=True)
    ang_img   = gaussian(ang_img,   2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)
    return q_img, ang_img, width_img


# ---------------- Visualization (3 columns) ----------------
def plot_results(fig, rgb_img, grasp_q_img, grasp_angle_img,
                 no_grasps=1, grasp_width_img=None, save_path=None):
    gs = detect_grasps(grasp_q_img, grasp_angle_img,
                       width_img=grasp_width_img, no_grasps=no_grasps)
    plt.clf()
    ax = fig.add_subplot(1, 3, 1); ax.imshow(rgb_img); ax.set_title('RGB'); ax.axis('off')
    ax = fig.add_subplot(1, 3, 2); ax.imshow(rgb_img); [g.plot(ax, color='r') for g in gs]; ax.set_title('Grasp'); ax.axis('off')
    ax = fig.add_subplot(1, 3, 3); plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1); ax.set_title('Q'); ax.axis('off'); plt.colorbar(plot, ax=ax, fraction=0.046)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f'Result saved -> {save_path}')


# ---------------- Main test ----------------
def main():
    Dataset = get_dataset("cornell")
    test_dataset = Dataset(ROOT_DATASET, output_size=OUTPUT_SIZE, ds_rotate=True,
                           random_rotate=False, random_zoom=False,
                           include_depth=True, include_rgb=True)
    indices = list(range(test_dataset.length))
    split = int(np.floor(0.01 * test_dataset.length))
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=4,
        sampler=torch.utils.data.SubsetRandomSampler(indices[split:]))

    ort_sess = ort.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider'])
    logger.info(f'ONNX loaded from {ONNX_PATH}')

    results = {'correct': 0, 'failed': 0}
    start_time = time.time()

    pbar = tqdm(val_loader, total=len(val_loader), desc='ONNX Testing')
    for idx, (x, y, didx, rot, zoom) in enumerate(pbar):
        x_np = x.numpy()
        torch.cuda.synchronize()
        t0 = time.time()
        ort_outs = ort_sess.run(None, {'input': x_np})
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / BATCH_SIZE * 1000

        q_np, cos_np, sin_np, w_np = ort_outs
        q_img, ang_img, w_img = post_process_output(q_np, cos_np, sin_np, w_np)

        for b in range(BATCH_SIZE):
            is_match = evaluation.calculate_iou_match(
                q_img[b], ang_img[b],
                val_loader.dataset.get_gtbb(didx[b], rot[b], zoom[b]),
                no_grasps=N_GRASPS, grasp_width=w_img[b], threshold=IOU_THR)
            results['correct' if is_match else 'failed'] += 1

        if idx < 20:
            fig = plt.figure(figsize=(12, 4))
            plot_results(fig,
                         rgb_img=test_dataset.get_rgb(didx[0], rot[0], zoom[0], normalise=False),
                         grasp_q_img=q_img[0],
                         grasp_angle_img=ang_img[0],
                         grasp_width_img=w_img[0],
                         save_path=os.path.join(SAVE_DIR, f'result_{idx:04d}.png'))
            plt.close(fig)

        pbar.set_postfix({'IoU': f"{results['correct']/(sum(results.values()) or 1):.3f}",
                          'ms': f'{elapsed:.1f}'})

    total = results['correct'] + results['failed']
    avg_iou = results['correct'] / total if total else 0.0
    avg_ms = (time.time() - start_time) / total * 1000
    logger.info(f'IoU Results: {results["correct"]}/{total} = {avg_iou:.4f}')
    logger.info(f'Average ONNX inference time per image: {avg_ms:.2f} ms')


if __name__ == '__main__':
    main()