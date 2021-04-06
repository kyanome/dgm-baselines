
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_reconstrunction(p, q, x):
    """
    reconstruct image given input observation x
    """
    height, width = x.shape[1:]
    x = x.reshape(x.size(0), -1)
    with torch.no_grad():
        # infer and sampling z using inference model q `.sample()` method
        z = q.sample({"x": x}, return_all=False)
        
        # reconstruct image from inferred latent variable z using Generator model p `.sample_mean()` method
        recon_batch = p.sample_mean(z).view(-1, 1, height, width)
        
        # concatenate original image and reconstructed image for comparison
        comparison = torch.cat([x.view(-1, 1, height, width), recon_batch]).cpu()
        return comparison
    
def plot_multimodal_reconstrunction(q, p_x, x, y):
    height, width = x.shape[1:]
    x = x.reshape(x.size(0), -1)
    with torch.no_grad():
        # infer from x and y
        z = q.sample({"x": x, "y": y}, return_all=False)
        # generate image from latent variable
        recon_batch = p_x.sample_mean(z).view(-1, 1, height, width)
        comparison = torch.cat([x.view(-1, 1, height, width), recon_batch]).cpu()
        return comparison

def plot_image_from_latent(p, z_sample, x):
    """
    generate new image given latent variable z
    """
    height, width = x.shape[1:]
    x = x.reshape(x.size(0), -1)
    with torch.no_grad():
        # generate image from latent variable z using Generator model p `.sample_mean()` method
        sample = p.sample_mean({"z": z_sample}).view(-1, 1, height, width).cpu()
        return sample
    
def plot_latent_space(q, z_dim, dataset, device):
    fig = plt.figure()
    z_list = []
    y_list = []
    with torch.no_grad():
        for i, (_, _, x, _, y) in enumerate(dataset):
            x = x.reshape(x.size(0), -1).to(device)
            z = q.sample_mean({"x": x})
            z_list.append(z.detach().cpu().numpy())
            y_list.append(y.numpy())
            if i == 2:
                break
        z = np.array(z_list).reshape(-1, z_dim)
        y = np.array(y_list).flatten()
        z_reduced = TSNE(n_components=2, random_state=0).fit_transform(z)
        plt.scatter(z_reduced[:,0],z_reduced[:,1],c=y,cmap=plt.cm.get_cmap('jet', 2))
        plt.colorbar()
        fig.canvas.draw()
        plot_image = fig.canvas.renderer._renderer
        return np.array(plot_image).transpose(2, 0, 1)

def plot_image_latent_space(q, z_dim, dataset, device):
    fig = plt.figure()
    z_list = []
    y1_list = []
    y2_list = []
    with torch.no_grad():
        for i, (_, _, x, y1, y2) in enumerate(dataset):
            x = x.reshape(x.size(0), -1).to(device)
            z = q.sample_mean({"x": x})
            z_list.append(z.detach().cpu().numpy())
            y2_list.append(y2.numpy())
            if i == 2:
                break
        z = np.array(z_list, dtype="object")
        y = np.array(y2_list, dtype="object")
        z = np.concatenate([z[0], z[1], z[2]])
        y = np.concatenate([y[0], y[1], y[2]])
        z_reduced = TSNE(n_components=2, random_state=0).fit_transform(z)
        plt.scatter(z_reduced[:,0],z_reduced[:,1],c=y)
        plt.colorbar()
        fig.canvas.draw()
        plot_image = fig.canvas.renderer._renderer
        return np.array(plot_image).transpose(2, 0, 1)

def plot_multimodal_latent_space(q, z_dim, dataset, y_ulabel, device):
    fig = plt.figure()
    z_list = []
    y1_list = []
    y2_list = []
    with torch.no_grad():
        for i, (_, _, x, y1, y2) in enumerate(dataset):
            x = x.reshape(x.size(0), -1).to(device)
            z = q.sample_mean({"x": x, "y": torch.eye(y_ulabel)[y2].to(device)})
            z_list.append(z.detach().cpu().numpy())
            y2_list.append(y2.numpy())
            if i == 2:
                break
        z = np.array(z_list, dtype="object")
        y = np.array(y2_list, dtype="object")
        z = np.concatenate([z[0], z[1], z[2]])
        y = np.concatenate([y[0], y[1], y[2]])
        z_reduced = TSNE(n_components=2, random_state=0).fit_transform(z)
        plt.scatter(z_reduced[:,0],z_reduced[:,1],c=y)
        plt.colorbar()
        fig.canvas.draw()
        plot_image = fig.canvas.renderer._renderer
        return np.array(plot_image).transpose(2, 0, 1)


def plot_reconstrunction_missing_label_modality(q_x, p_x, x):
    height, width = x.shape[1:]
    x = x.reshape(x.size(0), -1)
    with torch.no_grad():
        # infer from x (image modality) only
        z = q_x.sample({"x": x}, return_all=False)
        # generate image from latent variable
        recon_batch = p_x.sample_mean(z).view(-1, 1, height, width)
        comparison = torch.cat([x.view(-1, 1, height, width), recon_batch]).cpu()
        return comparison
    
def plot_image_from_label(q_y, p_x, x, y):
    height, width = x.shape[1:]
    x = x.reshape(x.size(0), -1)
    with torch.no_grad():
        #x_all = [x.view(-1, 1, height, width)]
        # infer from y (label modality) only
        z = q_y.sample({"y": y}, return_all=False)
        # generate image from latent variable
        recon_batch = p_x.sample_mean(z).view(-1, 1, height, width)
        comparison = torch.cat([x.view(-1, 1, height, width), recon_batch]).cpu()
        #x_all.append(recon_batch)
        #comparison = torch.cat(x_all).cpu()
        return comparison