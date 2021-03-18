import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from pixyz.losses import KullbackLeibler
from pixyz.models import VAE
from pixyz.distributions import Normal
from .model import GeneratorX, GeneratorY, InferenceX, InferenceY, Inference
import datetime
from tensorboardX import SummaryWriter
from baselines.common.utils import plot_multimodal_latent_space, plot_multimodal_reconstrunction, plot_image_from_latent, plot_reconstrunction_missing_label_modality, plot_image_from_label, plot_image_latent_space

class JMVAE():
    def __init__(self, z_dim, dataset,  batch_size, epochs=50):
        self._epochs = epochs
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = dataset.train_loader
        self.test_loader = dataset.test_loader
        _x, _y = iter(self.test_loader).next()
        self._x = _x.to(self._device)
        self._y = torch.eye(10)[_y].to(self._device)
        self._batch_size = batch_size
        self._x_dim = self._x.reshape(self._batch_size, -1).size(1)
        self._y_dim = 10
        self._z_dim = z_dim
        # Tensorboard
        dt_now = datetime.datetime.now()
        exp_time = dt_now.strftime('%Y%m%d_%H:%M:%S')
        nb_name = 'jmvae'
        self.writer = SummaryWriter("results/runs/{}/".format(nb_name) + exp_time)
        self.z_sample = 0.5 * torch.randn(64, z_dim).to(self._device)
        
        self.init_model()
        
    def init_model(self):
        # Model and Loss
        self.p_x = GeneratorX(self._x_dim, self._z_dim).to(self._device)
        self.p_y = GeneratorY(self._y_dim, self._z_dim).to(self._device)
        p = self.p_x * self.p_y
        self.q_x = InferenceX(self._x_dim, self._z_dim).to(self._device)
        self.q_y = InferenceY(self._y_dim, self._z_dim).to(self._device)
        q = Inference(self._x_dim, self._y_dim, self._z_dim).to(self._device)
        prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["z"], features_shape=[self._z_dim], name="p_{prior}").to(self._device)
        kl = KullbackLeibler(q, prior)
        kl_x = KullbackLeibler(q, self.q_x)
        kl_y = KullbackLeibler(q, self.q_y)
        regularizer = kl + kl_x + kl_y
        self._model = VAE(q, p, other_distributions=[self.q_x, self.q_y], regularizer=regularizer, optimizer=optim.Adam, optimizer_params={"lr":1e-3})
        self.q = q
    
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
        for x, y in tqdm(self.train_loader):
            x = x.reshape(x.size(0), -1).to(self._device)
            y = torch.eye(10)[y].to(self._device)       
            loss = self._model.train({"x": x, "y": y})
            train_loss += loss
        
        train_loss = train_loss * self._batch_size / len(self.train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss
    
    def to_tensorboard(self, train_loss, test_loss, epoch):
        recon = plot_multimodal_reconstrunction(self.q, self.p_x, self._x[:8], self._y[:8])
        recon_from_label = plot_image_from_label(self.q_y, self.p_x, self._x[:8], self._y[:8])
        recon_from_image = plot_reconstrunction_missing_label_modality(self.q_x, self.p_x, self._x[:8])
        multi_modal_latent_space = plot_multimodal_latent_space(self.q, self._z_dim, self.test_loader, self._device)
        image_latent_space = plot_image_latent_space(self.q_x, self._z_dim, self.test_loader, self._device)
        self.writer.add_scalar('train_loss', train_loss.item(), epoch)
        self.writer.add_scalar('test_loss', test_loss.item(), epoch)   
        self.writer.add_images('reconstrunction_from_label', recon, epoch)
        self.writer.add_images('reconstrunction_from_image_and_label', recon_from_label, epoch)
        self.writer.add_images('reconstrunction_from_image', recon_from_image, epoch)
        self.writer.add_images('image_latent_space', image_latent_space.reshape(1, 4, 480, 640), epoch)
        self.writer.add_images('multimodal_latent_space', multi_modal_latent_space.reshape(1, 4, 480, 640), epoch)
        
        
    

