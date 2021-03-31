import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
import sys
sys.path.append("../")
from utils import my_makedirs, get_dir_name, get_images, get_bbox, save_images, get_train_images

    
def main():
    data_path = "/media/keita/Seagate Expansion Drive/data/contactdb_data"
    save_path = "../data"
    intention_names = get_dir_name(data_path)
    images_names = ["thermal_images", "rgb_images", "depth_images"]

    for i_name in intention_names:
        print(i_name)
        object_names = get_dir_name(os.path.join(data_path, i_name))
        i_name_flag = "use" if "use" in i_name else "handoff"
        for o_name in object_names:
            print(o_name)
            my_makedirs(os.path.join(save_path, i_name_flag, o_name, images_names[0]))
            my_makedirs(os.path.join(save_path, i_name_flag, o_name, "normal_thermal_images"))
            my_makedirs(os.path.join(save_path, i_name_flag, o_name, images_names[1]))
            my_makedirs(os.path.join(save_path, i_name_flag, o_name, images_names[2]))
            thermal_imgs, mask_imgs, rgb_imgs, depth_imgs = get_images(os.path.join(data_path, i_name, o_name), images_names)
            save_images(thermal_imgs, mask_imgs, rgb_imgs, depth_imgs, os.path.join(save_path, i_name_flag, o_name))
    

def annotation():
    dir_path = "/home/keita/Research/dgm-baselines/baselines/common/dataset/contactdb/data"
    images_names = ["thermal_images", "rgb_images", "depth_images"]
    thermal_list, rgb_list, depth_list, object_labels, intention_labels = [], [], [], [], []
    
    handoff_object_names = get_dir_name(os.path.join(dir_path, "handoff"))
    for h_name in handoff_object_names:
        thermal_imgs, rgb_imgs, depth_imgs = get_train_images(os.path.join(dir_path, "handoff", h_name), images_names)
        if len(thermal_imgs) == 0:
            pass
        else:
            thermal_list.append(thermal_imgs)
            rgb_list.append(rgb_imgs)
            depth_list.append(depth_imgs)
            object_labels.append([h_name for i in range(len(thermal_imgs))])
            intention_labels.append(["handoff" for i in range(len(thermal_imgs))])
    
    use_object_names = get_dir_name(os.path.join(dir_path, "use"))
    for u_name in use_object_names:
        thermal_imgs, rgb_imgs, depth_imgs = get_train_images(os.path.join(dir_path, "use", u_name), images_names)
        if len(thermal_imgs) == 0:
            pass
        else:
            thermal_list.append(thermal_imgs)
            rgb_list.append(rgb_imgs)
            depth_list.append(depth_imgs)
            object_labels.append([u_name for i in range(len(thermal_imgs))])
            intention_labels.append(["use" for i in range(len(thermal_imgs))])
            
    thermal_list = list(chain.from_iterable(thermal_list))
    rgb_list = list(chain.from_iterable(rgb_list))
    depth_list = list(chain.from_iterable(depth_list))
    object_labels = list(chain.from_iterable(object_labels))
    intention_labels = list(chain.from_iterable(intention_labels))
    # convert list to array
    thermal_list = np.array(thermal_list)
    rgb_list = np.array(rgb_list)
    depth_list = np.array(depth_list)
    object_labels = np.array(object_labels)
    intention_labels = np.array(intention_labels)
    # save array
    my_makedirs(os.path.join(dir_path,"train"))
    np.save(os.path.join(dir_path,"train", 'thermal') , thermal_list)
    np.save(os.path.join(dir_path,"train", 'rgb'), rgb_list)
    np.save(os.path.join(dir_path,"train", 'depth'), depth_list)
    np.save(os.path.join(dir_path,"train", 'object_label'), object_labels)
    np.save(os.path.join(dir_path,"train", 'intention_label'), intention_labels)
    

if __name__ == '__main__':
    main()
    
