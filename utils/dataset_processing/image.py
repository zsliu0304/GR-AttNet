import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from skimage.transform import rotate, resize

warnings.filterwarnings("ignore", category=UserWarning)


class Image:
    """
    Wrapper around an image with some convenient functions.
    """
    def __init__(self, img: np.ndarray):
        self.img = img

    def __getattr__(self, attr: str):
        return getattr(self.img, attr)

    @classmethod
    def from_file(cls, fname: str):
        return cls(imread(fname))

    def copy(self) -> 'Image':
        return self.__class__(self.img.copy())

    def crop(self, top_left: tuple, bottom_right: tuple, resize: tuple = None):
        self.img = self.img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        if resize is not None:
            self.resize(resize)

    def cropped(self, *args, **kwargs) -> 'Image':
        i = self.copy()
        i.crop(*args, **kwargs)
        return i

    def normalise(self):
        self.img = self.img.astype(np.float32) / 255.0
        self.img -= self.img.mean()

    def resize(self, shape: tuple):
        if self.img.shape == shape:
            return
        self.img = resize(self.img, shape, preserve_range=True).astype(self.img.dtype)

    def resized(self, *args, **kwargs) -> 'Image':
        i = self.copy()
        i.resize(*args, **kwargs)
        return i

    def rotate(self, angle: float, center: tuple = None):
        if center is not None:
            center = (center[1], center[0])
        self.img = rotate(self.img, angle / np.pi * 180, center=center, mode='symmetric', preserve_range=True).astype(self.img.dtype)

    def rotated(self, *args, **kwargs) -> 'Image':
        i = self.copy()
        i.rotate(*args, **kwargs)
        return i

    def show(self, ax=None, **kwargs):
        if ax:
            ax.imshow(self.img, **kwargs)
        else:
            plt.imshow(self.img, **kwargs)
            plt.show()

    def zoom(self, factor: float):
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2
        orig_shape = self.img.shape
        self.img = self.img[sr:self.img.shape[0] - sr, sc: self.img.shape[1] - sc].copy()
        self.img = resize(self.img, orig_shape, mode='symmetric', preserve_range=True).astype(self.img.dtype)

    def zoomed(self, *args, **kwargs) -> 'Image':
        i = self.copy()
        i.zoom(*args, **kwargs)
        return i


class DepthImage(Image):
    def __init__(self, img: np.ndarray):
        super().__init__(img)

    @classmethod
    def from_pcd(cls, pcd_filename: str, shape: tuple, default_filler: float = 0, index: int = None):
        img = np.full(shape, default_filler, dtype=np.float32)
        with open(pcd_filename) as f:
            for l in f.readlines():
                ls = l.split()
                if len(ls) != 5:
                    continue
                try:
                    float(ls[0])
                except ValueError:
                    continue

                i = int(ls[4])
                r = i // shape[1]
                c = i % shape[1]

                if index is None:
                    x = float(ls[0])
                    y = float(ls[1])
                    z = float(ls[2])
                    img[r, c] = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                else:
                    img[r, c] = float(ls[index])

        return cls(img / 1000.0)

    @classmethod
    def from_tiff(cls, fname: str):
        return cls(imread(fname))

    def inpaint(self, missing_value: float = 0):
        self.img = cv2.copyMakeBorder(self.img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (self.img == missing_value).astype(np.uint8)
        scale = np.abs(self.img).max()
        self.img = cv2.inpaint((self.img / scale).astype(np.float32), mask, 1, cv2.INPAINT_NS) * scale
        self.img = self.img[1:-1, 1:-1]

    def gradients(self):
        grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_DEFAULT)
        grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return DepthImage(grad_x), DepthImage(grad_y), DepthImage(grad)

    def normalise(self):
        self.img = np.clip((self.img - self.img.mean()), -1, 1)


class WidthImage(Image):
    def zoom(self, factor: float):
        super().zoom(factor)
        self.img /= factor

    def normalise(self):
        self.img = np.clip(self.img, 0, 150.0) / 150.0