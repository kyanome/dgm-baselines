from baselines import VAE, MVAE, JMVAE
from baselines.common.dataset.mnist import MNIST
from baselines.common.dataset.contactdb import ContactDB

def main():
    epochs = 100
    batch_size = 64
    z_dim = 64
    dataset = ContactDB(batch_size)
    jmvae = JMVAE(z_dim, dataset, batch_size, epochs)
    #mvae = MVAE(z_dim, dataset, batch_size, epochs)
    #vae = VAE(z_dim, dataset, batch_size, epochs)
    
    # Evaluation
    jmvae.fit()
    #mvae.fit()
    #vae.fit()
    

if __name__ == '__main__':
    main()
    