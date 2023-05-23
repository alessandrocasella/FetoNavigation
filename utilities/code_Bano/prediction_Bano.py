import logging
import math
import os
import sys
from itertools import product

import cv2
import kornia.metrics
import numpy as np
import monai
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from monai.data import decollate_batch, ThreadDataLoader, CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
    AsChannelFirstd, LoadImaged,
    ScaleIntensityRanged, RandAffined, RandSpatialCropd, RandAxisFlipd, RandGaussianNoised, TorchVisiond, EnsureTyped,
    ToDeviced, AddChanneld, AsDiscreted, RandGaussianSmoothd, LabelFilterd, LabelFilter, Rotate90d, Flipd, Rotated
)
from monai.utils import set_determinism
from tqdm import tqdm


def main(exp, val):
    monai.config.print_config()
    device = torch.device("cuda")

    # create a temporary directory and 40 random image, mask pairs
    pat = ['anon024_CLIP01']
    images_val = [os.path.join('../../final_dataset', y, 'images', x) for y in pat for x in sorted(os.listdir(os.path.join(
        '../../final_dataset', y, 'images')))]

    val_dict = [{"image": image_name, "filename": image_name} for image_name in images_val]

    val_imtrans = Compose([
        LoadImaged(["image"], reader='pilreader', ensure_channel_first=True, image_only=True),
        Rotated(["image"], angle=-math.pi/2),
        Flipd(["image"], 0),
        ScaleIntensityRanged("image",0, 255, 0, 1),
        EnsureTyped(["image"],),
        ToDeviced(["image"],device=device)
    ])
    print(val_dict)
    print(val_imtrans)
    val_ds = CacheDataset(val_dict, val_imtrans, cache_rate=1.0, copy_cache=True, num_workers=4)
    val_loader = ThreadDataLoader(val_ds, batch_size=1, num_workers=0)
    post_trans = Compose([EnsureType(), Activations(softmax=True)])
    post_trans2 = Compose([AsDiscrete(argmax=True), LabelFilter(1)])
    post_transo = Compose([AsDiscrete(argmax=True), LabelFilter(1)])
    # create UNet, DiceLoss and Adam optimizer

    model = (torch.load(exp))
    model.eval()
    with torch.no_grad():

        for i, batch_data in enumerate(tqdm(val_loader)):
            val_images =  batch_data["image"].to(device)
            filename = os.path.basename(batch_data["filename"][0])
            save_path = os.path.join(os.getcwd(), 'bano_pred' )

            roi_size = (256, 256)
            sw_batch_size = 16
            val_pred = sliding_window_inference(val_images, roi_size, sw_batch_size, model,overlap=0.75, sw_device=device, device=device)
            val_outputs = [post_trans2(post_trans(i)) for i in decollate_batch(val_pred)]
            save_out = [post_transo(post_trans(i)) for i in decollate_batch(val_pred)]
            save_out = save_out[0]*255.
            save_out = np.squeeze(np.uint8(save_out.cpu().numpy()))
            cv2.imwrite(os.path.join(save_path,filename), save_out)



if __name__ == "__main__":
    path_bano_pred= os.path.join(os.getcwd(), 'bano_pred')
    if not os.path.exists(path_bano_pred):
        # print('creating new panorama image folder')
        os.makedirs(path_bano_pred)


    dirVideo = sorted(os.listdir(os.path.join(os.getcwd(), '../../final_dataset')))
    exp = os.path.join(os.getcwd(), '../../resnetfinal.pth')
    val_list = dirVideo
    set_determinism(seed=1990)
    main(exp, val_list)






