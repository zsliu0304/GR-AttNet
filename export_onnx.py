# export_onnx.py
import torch
import logging
from network import GRAttNet

# ===== Config =====
PTH_WEIGHT   = './checkpoints/best_iou.pth'   # trained weights
OUTPUT_ONNX  = './grattnet.onnx'              # export path
INPUT_SIZE   = 224                            # same as training
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE   = 1                              # dynamic batch can be set later
# ==================

def main():
    # 1. Build model and load weights
    model = GRAttNet().to(DEVICE)
    state_dict = torch.load(PTH_WEIGHT, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f'Loaded checkpoint from {PTH_WEIGHT}')

    # 2. Create dummy input
    dummy_input = torch.randn(BATCH_SIZE, 4, INPUT_SIZE, INPUT_SIZE, device=DEVICE)

    # 3. Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_ONNX,
        input_names=['input'],          # matches ort_sess.run(None, {'input': x_np})
        output_names=['pos', 'cos', 'sin', 'width'],
        opset_version=13,               # >=11 is fine
        do_constant_folding=True,       # fold constants to shrink size
        dynamic_axes=None               # set here if dynamic batch/size needed
    )
    logger.info(f'ONNX exported -> {OUTPUT_ONNX}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    logger = logging.getLogger(__name__)
    main()