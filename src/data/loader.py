import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Module-level placeholders for normalization
default_mean = None
default_std = None

def robust_image_loader(image_dir, mask_dir, mean, std, target_size=(64, 64), mask_suffix='_mask.png'):
    """
    Loads and normalizes images and masks based on provided mean/std.
    """
    images, masks = [], []
    for img_name in sorted(os.listdir(image_dir)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, base + mask_suffix)
        if not os.path.exists(mask_path):
            print(f"Skipping {img_name}: mask not found")
            continue
        img = img_to_array(
            load_img(img_path, target_size=target_size)
        ).astype('float32') / 255.0
        img = (img - mean) / (std + 1e-6)
        msk = img_to_array(
            load_img(mask_path, target_size=target_size, color_mode='grayscale')
        ).astype('float32') / 255.0
        images.append(img)
        masks.append(msk)
    return np.array(images), np.array(masks)
