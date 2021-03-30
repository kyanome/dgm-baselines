import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def get_dir_name(path):
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
    return files_dir

def get_images(path, img_names):
    temp_images = glob.glob(os.path.join(path, img_names[0], '*.png'))
    thermal_images = [img_name for img_name in temp_images if not "mask" in img_name]
    mask_images = [img_name for img_name in temp_images if "mask" in img_name]
    rgb_images = glob.glob(os.path.join(path, img_names[1], '*registered*.png'))
    depth_images = glob.glob(os.path.join(path, img_names[2], '*registered*.png'))
    thermal_images.sort()
    mask_images.sort()
    rgb_images.sort()
    depth_images.sort()
    return thermal_images, mask_images, rgb_images, depth_images

def get_train_images(path, img_names):
    thermal_images = glob.glob(os.path.join(path, img_names[0], '*.png'))
    rgb_images = glob.glob(os.path.join(path, img_names[1], '*.png'))
    depth_images = glob.glob(os.path.join(path, img_names[2], '*.png'))
    thermal_images = [name.replace("../", "") for name in thermal_images]
    rgb_images = [name.replace("../", "") for name in rgb_images]
    depth_images = [name.replace("../", "") for name in depth_images]
    thermal_images.sort()
    rgb_images.sort()
    depth_images.sort()
    return thermal_images, rgb_images, depth_images

def get_bbox(mask):
    maskx = np.any(mask, axis=0)
    masky = np.any(mask, axis=1)
    x1 = np.argmax(maskx)
    y1 = np.argmax(masky)
    x2 = len(maskx) - np.argmax(maskx[::-1])
    y2 = len(masky) - np.argmax(masky[::-1])
    return y1, y2, x1, x2


def save_images(ther_img, mask_img, rgb_img, depth_img, path):
    N = len(ther_img)
    for i in range(N):
        try:
            thermal = np.array(Image.open(ther_img[i]).convert('L'))
            mask = np.array(Image.open(mask_img[i]).convert('L'))
            rgb = np.array(Image.open(rgb_img[i]))
            depth = np.array(Image.open(depth_img[i]))
            thermal = thermal * mask
            rgb = rgb * np.repeat(mask[...,None],3,axis=2)
            depth = depth * mask
            y1, y2, x1, x2 = get_bbox(mask)
            buffer_y = int((256 - (y2 - y1))/2)
            buffer_x = int((320 - (x2 - x1))/2)
            mask = mask[y1-buffer_y:y2+buffer_y, x1-buffer_x:x2+buffer_x]
            thermal = thermal[y1-buffer_y:y2+buffer_y, x1-buffer_x:x2+buffer_x] 
            rgb = rgb[y1-buffer_y:y2+buffer_y, x1-buffer_x:x2+buffer_x] 
            depth = depth[y1-buffer_y:y2+buffer_y, x1-buffer_x:x2+buffer_x]
            if 6750 < thermal.flatten().sum():
                Image.fromarray(thermal).save(os.path.join(path, "thermal_images", "{:0=2}_thermal.png".format(i)))
                Image.fromarray(rgb).save(os.path.join(path, "rgb_images", "{:0=2}_rgb.png".format(i)))
                Image.fromarray(depth).save(os.path.join(path, "depth_images", "{:0=2}_depth.png".format(i)))
        except:
            pass    