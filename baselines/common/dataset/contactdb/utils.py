import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps

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
    img_size = (640, 512)
    w, h = 320, 256
    sigmoid_k = 10
    max_frac = 0.75
    N = len(ther_img)
    for i in range(N):
        try:
            thermal = Image.open(ther_img[i])
            mask = Image.open(mask_img[i])
            rgb = Image.open(rgb_img[i])
            depth = Image.open(depth_img[i])
            
            m = np.asarray(mask)
            mask_pix = np.where(m)
            try:
                x_min, y_min = min(mask_pix[1]), min(mask_pix[0])
                x_max, y_max = max(mask_pix[1]), max(mask_pix[0])
                cx, cy = (x_min+x_max) / 2, (y_min+y_max) / 2
            except:
                print (img_name)
                outf.write(img_name+'\n')
                
            coords = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
            cropped_thermal = thermal.crop(coords)
            cropped_rgb = rgb.crop(coords)
            cropped_depth = depth.crop(coords)
            cropped_mask = mask.crop(coords)
            
            _cropped_thermal = np.array(cropped_thermal)
            _cropped_mask = np.array(cropped_mask)
            apply_mask_thermal = _cropped_thermal * np.repeat(_cropped_mask[...,None],3,axis=2)
            apply_mask_thermal = ImageOps.invert(Image.fromarray(apply_mask_thermal).convert('L'))
            apply_mask_thermal = np.array(apply_mask_thermal)
            apply_mask_thermal[apply_mask_thermal==255] = 0
            if apply_mask_thermal.flatten().sum() >= 60000:
                apply_mask_thermal = Image.fromarray(apply_mask_thermal).convert("L")
                apply_mask_thermal.save(os.path.join(path, "thermal_images", "{:0=2}_thermal.png".format(i)))
                cropped_thermal.save(os.path.join(path, "normal_thermal_images", "{:0=2}_thermal.png".format(i)))
                cropped_rgb.save(os.path.join(path, "rgb_images", "{:0=2}_rgb.png".format(i)))
                cropped_depth.save(os.path.join(path, "depth_images", "{:0=2}_depth.png".format(i)))
        except:
            pass    