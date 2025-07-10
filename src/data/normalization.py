import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def compute_mean_std(image_dir, mask_dir, target_size=(64, 64), mask_suffix='_mask.png'):
    """
    Computes per-channel mean and standard deviation over all images (excluding masks).
    """
    imgs = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        base = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_dir, base + mask_suffix)
        if not os.path.exists(mask_path):
            continue
        arr = img_to_array(
            load_img(os.path.join(image_dir, fname), target_size=target_size)
        ).astype('float32') / 255.0
        imgs.append(arr)
    X = np.stack(imgs, axis=0)
    mean = np.mean(X, axis=(0, 1, 2), keepdims=True)
    std = np.std(X, axis=(0, 1, 2), keepdims=True)
    return mean, std
