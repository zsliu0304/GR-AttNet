import glob
import os
import numpy as np
from utils.dataset_processing import grasp, image
from .grasp_data import GraspDatasetBase


class MulticornellDataset(GraspDatasetBase):
    """
    Dataset wrapper for the new Cornell-like data triplets:
        rgb_XXXX.jpg
        depth_XXXX.png
        rgb_XXXX_annotations.txt
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        super(MulticornellDataset, self).__init__(**kwargs)

        # 1. Anchor on annotation files to find the complete triplets
        self.anno_files = glob.glob(os.path.join(file_path, '*_annotations.txt'))
        if not self.anno_files:
            raise FileNotFoundError('No *_annotations.txt found. Check path: {}'.format(file_path))
        self.anno_files.sort()

        self.length = len(self.anno_files)
        if ds_rotate:
            self.anno_files = (self.anno_files[int(self.length * ds_rotate):] +
                               self.anno_files[:int(self.length * ds_rotate)])

        # 2. Derive RGB and depth paths
        self.rgb_files   = [f.replace('_annotations.txt', '.jpg') for f in self.anno_files]
        self.depth_files = [f.replace('_annotations.txt', '.png').replace('rgb_', 'depth_')
                            for f in self.anno_files]

        # 3. Quick sanity check (optional but useful)
        for triplet in zip(self.rgb_files, self.depth_files, self.anno_files):
            for f in triplet:
                if not os.path.isfile(f):
                    raise FileNotFoundError('Missing file: {}'.format(f))

    # ------------------------------------------------------------------
    def _get_crop_attrs(self, idx):
        """Return grasp center and top-left corner for cropping."""
        gtbbs  = grasp.GraspRectangles.load_from_cornell_file(self.anno_files[idx])
        center = gtbbs.center
        left   = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top    = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        """Load and transform ground-truth grasp rectangles."""
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.anno_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        """Load, rotate, crop, normalise and resize depth image."""
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])  # internally handles PNG
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left),
                       (min(480, top + self.output_size),
                        min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        """Load, rotate, crop, normalise and resize RGB image."""
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left),
                     (min(480, top + self.output_size),
                      min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img