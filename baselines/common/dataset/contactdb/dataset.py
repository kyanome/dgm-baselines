import torch
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
from .utils import get_dir_name
from sklearn.preprocessing import LabelEncoder

class Dataset(data.Dataset):
    def __init__(self, dir_path):
        super().__init__()
        
        self.label_encoder = LabelEncoder()
        self.dir_path = dir_path
        self.rgb_paths = np.load(os.path.join(dir_path, "rgb.npy"))
        self.depth_paths = np.load(os.path.join(dir_path, "depth.npy"))
        self.thermal_paths = np.load(os.path.join(dir_path, "thermal.npy"))
        
        intention_label = np.load(os.path.join(dir_path, "intention_label.npy"))
        self.intention_label = self.label_encoder.fit_transform(intention_label)

        object_label = np.load(os.path.join(dir_path, "object_label.npy"))
        self.object_label = self.label_encoder.fit_transform(object_label)
        
        self.input_size = (64, 64)
        self.len = len(self.rgb_paths)
        
        
    def __len__(self):
        return self.len
    
    def read_image(self, path):
        image = Image.open(path)
        image = image.resize(self.input_size)
        image = np.array(image)
        #image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32)) / 255.
        return image

    def get_u_label(self):
        u_y1 = np.unique(self.object_label)
        u_y2 = np.unique(self.intention_label)
        return len(u_y1), len(u_y2)
    
    def __getitem__(self, index):
        rgb = self.rgb_paths[index]
        depth = self.depth_paths[index]
        thermal = self.thermal_paths[index]
        object_label = self.object_label[index]
        intention_label = self.intention_label[index]
    
        rgb_img = self.read_image(rgb)
        depth_img = self.read_image(depth)
        thermal_img = self.read_image(thermal)
        
        return rgb_img, depth_img, thermal_img, object_label, intention_label

class ContactDB():
    def __init__(self, batch_size=32):
        dataset = Dataset("/home/keita/Research/dgm-baselines/baselines/common/dataset/contactdb/data/train")
        n_samples = len(dataset) 
        train_size = int(len(dataset) * 0.8) 
        val_size = n_samples - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
        self.train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **kwargs)
        self.dataset = dataset

if __name__ == '__main__':
    contact_db = ContactDB()
    r, d, t, o, i = iter(contact_db.test_loader).next()
    import pdb; pdb.set_trace()
    