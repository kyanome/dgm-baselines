import torch
from torch import nn
from torch.nn import functional as F
from pixyz.distributions import Normal, Bernoulli

# inference model q(z|x)
class Inference(Normal):
    """
    parameterizes q(z | x)
    infered z follows a Gaussian distribution with mean 'loc', variance 'scale'
    z ~ N(loc, scale)
    """
    def __init__(self, x_dim=784, z_dim=64):
        super(Inference, self).__init__(var=["z"], cond_var=["x"], name="q")

        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)

    def forward(self, x):
        """
        given the observation x,
        return the mean and variance of the Gaussian distritbution
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    
# generative model p(x|z)    
class Generator(Bernoulli):
    """
    parameterizes the bernoulli(for MNIST) observation likelihood p(x | z)
    """
    def __init__(self, x_dim=784, z_dim=64):
        super(Generator, self).__init__(var=["x"], cond_var=["z"], name="p")

        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, x_dim)

    def forward(self, z):
        """
        given the latent variable z,
        return the probability of Bernoulli distribution
        """
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return {"probs": torch.sigmoid(self.fc3(h))}

