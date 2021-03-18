import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from pixyz.losses import LogProb, KullbackLeibler, Expectation as E
from pixyz.models import Model
from pixyz.distributions import Normal
from .model import Generator, Inference
import datetime
from tensorboardX import SummaryWriter
from baselines.common.utils import plot_latent_space, plot_reconstrunction, plot_image_from_latent


class VAE():
    def __init__(self, z_dim, dataset,  batch_size, epochs=50):
        self._epochs = epochs
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = dataset.train_loader
        self.test_loader = dataset.test_loader
        _x, _y = iter(self.test_loader).next()
        self._x = _x.to(self._device)
        self._y = _y.numpy()
        self._batch_size = batch_size
        self._x_dim = self._x.reshape(self._batch_size, -1).size(1)
        self._z_dim = z_dim
        # Model and Loss
        self.p = Generator().to(self._device)
        self.q = Inference().to(self._device)
        prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["z"], features_shape=[z_dim], name="p_{prior}").to(self._device)
        loss_fn = (KullbackLeibler(self.q, prior) - E(self.q, LogProb(self.p))).mean()
        self._model = Model(loss=loss_fn, distributions=[self.p, self.q], optimizer=optim.Adam, optimizer_params={"lr":1e-3})
        # Tensorboard
        dt_now = datetime.datetime.now()
        exp_time = dt_now.strftime('%Y%m%d_%H:%M:%S')
        nb_name = 'vae'
        self.writer = SummaryWriter("results/runs/{}/".format(nb_name) + exp_time)
        self.z_sample = 0.5 * torch.randn(64, z_dim).to(self._device)
    
    def fit(self):
        for epoch in range(1, self._epochs + 1):
            train_loss = self.train(epoch)
            test_loss = self.test(epoch)
            self.to_tensorboard(train_loss, test_loss, epoch)
        self.writer.close()
        
    def test(self, epoch):
        test_loss = 0
        for x, y in self.test_loader:
            x = x.reshape(x.size(0), -1).to(self._device)
            y = torch.eye(10)[y].to(self._device)
            loss = self._model.test({"x": x, "y": y})
            test_loss += loss

        test_loss = test_loss * self._batch_size / len(self.test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    def train(self, epoch):
        train_loss = 0
        for x, _ in tqdm(self.train_loader):
            x = x.reshape(x.size(0), -1).to(self._device)
            loss = self._model.train({"x": x})
            train_loss += loss
    
        train_loss = train_loss * self._batch_size / len(self.train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    def to_tensorboard(self, train_loss, test_loss, epoch):
        latent_space = plot_latent_space(self.q, self._z_dim, self.test_loader, self._device)
        recon = plot_reconstrunction(self.p, self.q, self._x[:8])
        sample = plot_image_from_latent(self.p, self.z_sample, self._x[:8])
        self.writer.add_scalar('train_loss', train_loss.item(), epoch)
        self.writer.add_scalar('test_loss', test_loss.item(), epoch)      
        self.writer.add_images('Image_from_latent', sample, epoch)
        self.writer.add_images('Image_reconstrunction', recon, epoch)
        self.writer.add_images('Latent_space', latent_space.reshape(1, 4, 480, 640), epoch)
        
        
    

