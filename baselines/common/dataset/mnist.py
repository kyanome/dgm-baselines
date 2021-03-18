
from torchvision import datasets, transforms
import torch

class MNIST():
    def __init__(self, batch_size=128):
        #wget www.di.ens.fr/~lelarge/MNIST.tar.gz
        #tar -zxvf MNIST.tar.gz
        transform = transforms.Compose([transforms.ToTensor()])
        kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
        train_set = datasets.MNIST('./baselines/common/dataset/', download=False, transform=transform, train=True)
        test_set = datasets.MNIST('./baselines/common/dataset/', download=False, transform=transform, train=False)
        self.train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, **kwargs)

