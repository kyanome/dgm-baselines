import torch
from torch import nn
from torch.nn import functional as F
from pixyz.distributions import Normal, Bernoulli,Categorical

# inference model q(z|x,y)
class Inference(Normal):
    def __init__(self, x_dim, y_dim, z_dim):
        super(Inference, self).__init__(var=["z"], cond_var=["x","y"], name="q")

        self.fc1 = nn.Linear(x_dim+y_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)

    def forward(self, x, y):
        h = F.relu(self.fc1(torch.cat([x, y], 1)))
        h = F.relu(self.fc2(h))        
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}
    

# inference model q(z|x)
class InferenceX(Normal):
    def __init__(self, x_dim, z_dim):
        super(InferenceX, self).__init__(var=["z"], cond_var=["x"], name="q")

        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))        
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}


# inference model q(z|y)
class InferenceY(Normal):
    def __init__(self, y_dim, z_dim):
        super(InferenceY, self).__init__(var=["z"], cond_var=["y"], name="q")

        self.fc1 = nn.Linear(y_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)

    def forward(self, y):
        h = F.relu(self.fc1(y))
        h = F.relu(self.fc2(h))        
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    
# generative model p(x|z)    
class GeneratorX(Bernoulli):
    def __init__(self, x_dim, z_dim):
        super(GeneratorX, self).__init__(var=["x"], cond_var=["z"], name="p")

        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, x_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return {"probs": torch.sigmoid(self.fc3(h))}


# generative model p(y|z)    
class GeneratorY(Categorical):
    def __init__(self, y_dim, z_dim):
        super(GeneratorY, self).__init__(var=["y"], cond_var=["z"], name="p")

        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, y_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return {"probs": F.softmax(self.fc3(h), dim=1)}