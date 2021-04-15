import numpy as np
from astropy.io import fits
from scipy import ndimage
from astropy.visualization import SqrtStretch, LogStretch

__all__ = ['img_proc', 'read_img']


RAW_SIZE = 7.5
COLOR = False
STRETCH = 'sqrt'
IMG_SIZE = 128

TARGET_SIZE = 3
TRANSLATE = 0.
ROTATE = False


def img_proc(img, target_size=TARGET_SIZE, raw_size=RAW_SIZE, translate=TRANSLATE, rotate=ROTATE):
    assert img.shape[-2] == img.shape[-1]
    assert translate >= 0
    if not isinstance(target_size, (float, int)):
        target_size = np.random.uniform(*target_size)
    if rotate:
        assert target_size * (1 + translate) * 2**0.5 < raw_size
        rotate_d = np.random.uniform(0., 360.)
        img = ndimage.rotate(img, rotate_d, axes=(-1, -2), reshape=False)
    else:
        assert target_size * (1 + translate) < raw_size
    s = img.shape[-1]
    translate = translate * target_size
    t_x = np.random.uniform(-translate, translate) if translate > 0 else 0
    t_y = np.random.uniform(-translate, translate) if translate > 0 else 0
    a = int(s * (raw_size - target_size + t_x) / (2 * raw_size))
    b = int(s * (raw_size - target_size + t_y) / (2 * raw_size))
    c = int(s * target_size / raw_size)
    return img[..., a:(a + c), b:(b + c)]


def read_img(path, color=COLOR, stretch=STRETCH, img_size=IMG_SIZE, return_label=True, atleast_3d=True, table=None, **kwargs):
    fits_file = fits.open(path)
    img = fits_file[0].data[::-1] if color else np.mean(fits_file[0].data, axis=0)
    img = img_proc(img, **kwargs)
    if stretch == 'linear':
        img = img - np.mean(img)
    elif stretch == 'sqrt':
        img = SqrtStretch()(img)
    elif stretch == 'log':
        img = LogStretch()(img)
    if img_size is not None:
        z = img_size / img.shape[-1]
        if img.ndim == 2:
            img = ndimage.zoom(img, (z, z))
            if atleast_3d:
                img = img[np.newaxis]
        elif img.ndim == 3:
            img = ndimage.zoom(img, (1, z, z))
        else:
            raise RuntimeError
    if return_label:
        assert table is not None
        return img, np.array((table.iloc[index]['p_cw'], table.iloc[index]['p_acw']))
    else:
        return img
