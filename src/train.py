import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import numpy as np
import tensorflow as tf
from .data.normalization import compute_mean_std
from .data.loader import robust_image_loader
from .loss.delaunay_tv import DelaunayTVLoss
from .metrics.segmentation_metrics import dice_coef, iou_metric
from .model.unet import build_radial_attention_unet
from .utils.mesh_sampler import sample_random_points


def main(config):
    # Unpack config dict
    image_dir = config['image_dir']
    mask_dir = config['mask_dir']
    target_size = tuple(config.get('target_size', (64,64)))
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 150)

    mean, std = compute_mean_std(image_dir, mask_dir, target_size)
    X, Y = robust_image_loader(image_dir, mask_dir, mean, std, target_size)
    x_tr, x_val, y_tr, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    H, W = X.shape[1], X.shape[2]
    pts = sample_random_points(H, W, num_points=config.get('num_points',64), seed=config.get('seed',42))

    model = build_radial_attention_unet(
        input_shape=(*target_size, 3),
        base_filters=config.get('base_filters',64),
        radial_levels=config.get('radial_levels',8)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.get('learning_rate',1e-3)),
        loss=DelaunayTVLoss(points=pts, lambda_del=config.get('lambda_del',5e-4)),
        metrics=[dice_coef, iou_metric]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping('val_loss', patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.8, patience=5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint('saved_models/best_model.keras', save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    results = model.evaluate(x_val, y_val, return_dict=True)
    print("Eval results:", results)

    y_pred = model.predict(x_val).ravel()
    y_true = y_val.ravel()
    mAP = average_precision_score(y_true, y_pred)
    print(f"mAP: {mAP:.4f}")

if __name__ == '__main__':
    import yaml
    with open('configs/default.yaml') as fp:
        cfg = yaml.safe_load(fp)
    main(cfg)
