import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim, H, output_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, H)
        self.linear2 = nn.Linear(H, output_dim)

    def forward(self, data):
        data = data.reshape(-1)
        data = F.relu(self.linear1(data))
        return F.relu(self.linear2(data))


class Decoder(nn.Module):
    def __init__(self, input_dim, H, output_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, H)
        self.linear2 = nn.Linear(H, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.sigmoid(self.linear2(x))


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.mu = None
        self.sigma = None
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_mu = nn.Linear(220, 6)
        self.encoder_log_sigma = nn.Linear(220, 6)

    def latent_samples(self, enc):
        self.mu = self.encoder_mu(enc)
        self.sigma = torch.exp(self.encoder_log_sigma(enc))
        std = torch.from_numpy(np.random.normal(0, 1, size=self.sigma.size())).float()
        return self.mu + self.sigma * Variable(std, requires_grad=False)

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        z_samples = self.latent_samples(encoder_outputs)
        return self.decoder(z_samples), self.mu, self.sigma

    ''' Taken from https://chrisorm.github.io/VAE-pyt.html '''

    def ELBO_loss(self, x_tilda, x, mu, logvar):

        BCE = F.binary_cross_entropy(x_tilda, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


if __name__ == '__main__':

    input_dim = 220 * 220  # image size
    epochs = 100
    dataset = utils.DATASET
    encoder = Encoder(input_dim, 500, 220)
    decoder = Decoder(6, 500, input_dim)
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0001)

    x_output = None
    for epoch in range(epochs):
        for data in dataset:
            inputs = Variable(torch.Tensor(data))
            optimizer.zero_grad()
            vae_output, mu, sigma = vae(inputs)
            inputs = inputs.reshape(-1)
            elbo_loss = vae.ELBO_loss(vae_output, inputs, mu, sigma)
            criterion_loss = criterion(vae_output, inputs) + elbo_loss
            criterion_loss.backward()
            optimizer.step()
            x_output = vae_output.reshape(220, 220).detach().numpy()


