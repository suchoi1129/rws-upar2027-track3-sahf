# UPAR Track 3: degradation augmentation for low-quality surveillance images. Applied to the full source image right after
# LoadImageFromFile, i.e. before the crop, during training only. The image stays uint8 RGB HxWx3, so the rest of the official
# pipeline is unchanged.
import cv2
import numpy as np

from ..builder import PIPELINES

_UP = {'bilinear': cv2.INTER_LINEAR, 'bicubic': cv2.INTER_CUBIC, 'nearest': cv2.INTER_NEAREST}


@PIPELINES.register_module()
class TopDownDegrade:
    """Four degradations, each applied independently with its own probability, in this order:
      low resolution  p = 0.25  downscale by f ~ U(0.35, 0.9) (area), upscale back with bilinear / bicubic / nearest (random)
      Gaussian blur   p = 0.15  kernel size 3 or 5 (random), sigma from the kernel size
      JPEG            p = 0.30  quality ~ U{20, ..., 75}
      Gaussian noise  p = 0.10  independent per pixel and channel, sigma ~ U(2, 8) on the 0-255 scale
    An image may receive several of them. Only the appearance changes, not the pose geometry."""

    P = dict(lowres=0.25, blur=0.15, jpeg=0.3, noise=0.1)

    def __call__(self, results):
        img = results['img']
        H, W = img.shape[:2]
        if np.random.rand() < self.P['lowres']:
            f = np.random.uniform(0.35, 0.9)
            up = _UP[np.random.choice(list(_UP))]
            small = cv2.resize(img, (max(4, int(W * f)), max(8, int(H * f))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (W, H), interpolation=up)
        if np.random.rand() < self.P['blur']:
            k = int(np.random.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        if np.random.rand() < self.P['jpeg']:
            q = int(np.random.randint(20, 76))
            ok, enc = cv2.imencode('.jpg', img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
        if np.random.rand() < self.P['noise']:
            n = np.random.randn(*img.shape) * np.random.uniform(2, 8)
            img = np.clip(np.rint(img.astype(np.float32) + n), 0, 255).astype(np.uint8)
        results['img'] = np.ascontiguousarray(img)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.P})'
