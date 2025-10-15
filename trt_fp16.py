# trt_fp16.py —— Generate FP16 TensorRT engine and benchmark (Windows / Linux)
import os
import time
import logging

import torch
import torch_tensorrt as trt
import numpy as np
from tqdm import tqdm

from network import GRAttNet
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from skimage.filters import gaussian

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ===================== Parameters =====================
PTH_WEIGHT   = './checkpoints/best_iou.pth'   # trained weights
OUTPUT_TRT   = './grattnet_fp16.trt'          # output TensorRT engine
INPUT_SHAPE  = (1, 4, 224, 224)               # NCHW
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NRUNS        = 200                            # number of runs for speed test
# ======================================================


def build_trt_fp16():
    """Load weights → trace → compile FP16 → save engine"""
    logger.info("Building TensorRT FP16 engine...")
    model = GRAttNet().to(DEVICE)
    model.load_state_dict(torch.load(PTH_WEIGHT, map_location=DEVICE))
    model.eval()

    example = torch.randn(*INPUT_SHAPE, device=DEVICE)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    trt_model = trt.compile(
        traced,
        inputs=[trt.Input(INPUT_SHAPE)],
        enabled_precisions={torch.float16},   # FP16 only
        device={"device_type": trt.DeviceType.GPU, "gpu_id": 0}
    )

    torch.jit.save(trt_model, OUTPUT_TRT)
    logger.info(f'TensorRT FP16 engine saved -> {OUTPUT_TRT}')
    return trt_model


def post_process_trt(q_img, cos_img, sin_img, width_img):
    """Post-process TensorRT outputs → numpy"""
    q_img     = q_img.squeeze(0)          # remove batch dim
    ang_img   = np.arctan2(sin_img, cos_img) / 2.0
    width_img = width_img.squeeze(0) * 150.0

    q_img     = gaussian(q_img,     2.0, preserve_range=True)
    ang_img   = gaussian(ang_img,   2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)
    return q_img, ang_img, width_img


def benchmark_trt(trt_model, dataloader, nruns=200):
    """Benchmark speed + simple IoU evaluation"""
    trt_model.eval()
    timings = []
    results = {'correct': 0, 'failed': 0}

    pbar = tqdm(dataloader, total=min(nruns, len(dataloader)), desc='TensorRT FP16 Benchmark')
    for idx, (x, y, didx, rot, zoom) in enumerate(pbar):
        if idx >= nruns:
            break
        x = x.to(DEVICE)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            ort_outs = trt_model(x)          # TensorRT forward
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / x.size(0) * 1000  # ms per image

        timings.append(elapsed)

        # Post-process & IoU (single image)
        q_np, cos_np, sin_np, w_np = [o.cpu().numpy() for o in ort_outs]
        q_img, ang_img, w_img = post_process_trt(q_np, cos_np, sin_np, w_np)

        is_match = evaluation.calculate_iou_match(
            q_img[0], ang_img[0],
            dataloader.dataset.get_gtbb(didx[0], rot[0], zoom[0]),
            no_grasps=2, grasp_width=w_img[0], threshold=0.25)
        results['correct' if is_match else 'failed'] += 1
        pbar.set_postfix({'IoU': f"{results['correct']/(sum(results.values()) or 1):.3f}",
                          'ms': f'{elapsed:.1f}'})

    avg_ms = np.mean(timings)
    logger.info(f'TensorRT FP16 average time: {avg_ms:.2f} ms '
                f'(±{np.std(timings):.2f} ms) | IoU: {results["correct"]}/{sum(results.values())} = {results["correct"]/(sum(results.values()) or 1):.4f}')


def main():
    # 1. Build engine
    trt_model = build_trt_fp16()

    # 2. Load 10% of test set for benchmarking
    Dataset = get_dataset("cornell")
    test_dataset = Dataset('D:/cornell', output_size=224, ds_rotate=True,
                           random_rotate=False, random_zoom=False,
                           include_depth=True, include_rgb=True)
    indices = list(range(test_dataset.length))
    split = int(np.floor(0.1 * test_dataset.length))
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=4,
        sampler=torch.utils.data.SubsetRandomSampler(indices[split:]))

    # 3. Benchmark speed + accuracy
    benchmark_trt(trt_model, val_loader, nruns=200)


if __name__ == '__main__':
    main()