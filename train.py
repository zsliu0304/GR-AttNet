"""
Train GR-AttNet.
"""
import os
import time
import logging
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from network import GRAttNet
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from skimage.filters import gaussian

# ---------- Basic Config ----------
ROOT_DATASET = 'D:/cornell'   # dataset root
OUTPUT_SIZE  = 224
BATCH_SIZE   = 8
EPOCHS       = 50
INIT_LR      = 1e-3
IOU_THR      = 0.25
SAVE_DIR     = './checkpoints'
LOG_FILE     = './train.log'
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='w'),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------- Post-processing ----------
def post_process_output(q_img, cos_img, sin_img, width_img):
    q_img     = q_img.cpu().numpy().squeeze()
    ang_img   = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    q_img     = gaussian(q_img,     2.0, preserve_range=True)
    ang_img   = gaussian(ang_img,   2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)
    return q_img, ang_img, width_img


# ---------- Training Epoch ----------
def train_one_epoch(network, loader, optimizer, device, epoch):
    network.train()
    running_loss = 0.0
    pbar = tqdm(loader, total=len(loader), desc=f'Epoch {epoch:02d} train')
    for step, (x, y, _, _, _) in enumerate(pbar, 1):
        x, y = x.to(device), [yy.to(device) for yy in y]

        loss_dict = network.compute_loss(x, y)
        loss      = loss_dict['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = running_loss / len(loader)
    logger.info(f'Epoch {epoch:02d} | Train Loss: {avg_loss:.4f}')
    return avg_loss


@torch.no_grad()
def validate_one_epoch(network, loader, device, epoch):
    network.eval()
    val_loss   = 0.0
    n_correct  = 0
    n_total    = 0

    for x, y, didx, rot, zoom in tqdm(loader, desc=f'Epoch {epoch:02d} val'):
        x, y = x.to(device), [yy.to(device) for yy in y]
        loss_dict = network.compute_loss(x, y)
        val_loss += loss_dict['loss'].item()

        q_out, ang_out, w_out = post_process_output(
            loss_dict['pred']['pos'],
            loss_dict['pred']['cos'],
            loss_dict['pred']['sin'],
            loss_dict['pred']['width'])

        is_match = evaluation.calculate_iou_match(
            q_out, ang_out,
            loader.dataset.get_gtbb(didx, rot, zoom),
            no_grasps=1, grasp_width=w_out, threshold=IOU_THR)

        n_correct += int(is_match)
        n_total   += 1

    avg_loss = val_loss / len(loader)
    iou      = n_correct / n_total if n_total else 0.0
    logger.info(f'Epoch {epoch:02d} | Val Loss: {avg_loss:.4f} | '
                f'Val IoU: {iou:.4f} ({n_correct}/{n_total})')
    return avg_loss, iou


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # ---- Dataset ----
    Dataset = get_dataset("cornell")
    dataset = Dataset(ROOT_DATASET,
                      output_size=OUTPUT_SIZE,
                      ds_rotate=True,
                      random_rotate=True,
                      random_zoom=True,
                      include_depth=True,
                      include_rgb=True)

    indices = list(range(dataset.length))
    split   = int(np.floor(0.9 * dataset.length))
    train_ids, val_ids = indices[:split], indices[split:]

    train_loader = DataLoader(dataset,
                              batch_size=BATCH_SIZE,
                              sampler=torch.utils.data.SubsetRandomSampler(train_ids),
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(dataset,
                              batch_size=1,
                              sampler=torch.utils.data.SubsetRandomSampler(val_ids),
                              num_workers=4, pin_memory=True)

    # ---- Model & Optimizer ----
    network = GRAttNet().to(device)
    optimizer = optim.Adam(network.parameters(), lr=INIT_LR)
    logger.info('Model built, start training...')

    # ---- Records ----
    best_iou   = 0.0
    train_losses, val_losses, val_ious = [], []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(network, train_loader, optimizer, device, epoch)
        val_loss, val_iou = validate_one_epoch(network, val_loader, device, epoch)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        # ---- Save Checkpoint ----
        ckpt_name = os.path.join(SAVE_DIR, f'epoch_{epoch:02d}_iou_{val_iou:.4f}.pth')
        torch.save(network.state_dict(), ckpt_name)
        if val_iou > best_iou:
            best_iou = val_iou
            best_ckpt = os.path.join(SAVE_DIR, 'best_iou.pth')
            torch.save(network.state_dict(), best_ckpt)
            logger.info(f'New best IoU: {best_iou:.4f} -> saved to {best_ckpt}')

        logger.info(f'Epoch {epoch:02d} finished in {time.time() - t0:.1f}s\n')

    # ---- Plot Curves ----
    epochs_arr = np.arange(1, len(val_ious) + 1)
    plt.figure()
    plt.plot(epochs_arr, val_ious, 'b', label='Validation IoU')
    plt.title('Validation IoU')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'val_iou.png'))

    plt.figure()
    plt.plot(epochs_arr, train_losses, 'b', label='Train Loss')
    plt.plot(epochs_arr, val_losses, 'r', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'loss_curve.png'))
    plt.show()


if __name__ == '__main__':
    main()