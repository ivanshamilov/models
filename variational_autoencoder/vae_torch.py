import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from PIL import Image

from torchvision import transforms
from torch.nn import Linear, Conv2d, ConvTranspose2d, Dropout, BatchNorm2d, LeakyReLU, ReLU
from typing import Tuple, List

from tqdm import tqdm


class Encoder(nn.Module):
  def __init__(self,
               input_dim: Tuple[int],
               encoder_conv_filters: List[int],
               encoder_conv_kernel_size: List[int],
               encoder_conv_strides: List[int],
               z_dim: int,
               use_batch_norm: bool = False,
               use_dropout: bool = False):
    super(Encoder, self).__init__()
    self.encoder_layers = [
      Conv2d(in_channels=input_dim[-1], 
          out_channels=encoder_conv_filters[0], 
          kernel_size=encoder_conv_kernel_size[0],
          stride=encoder_conv_strides[0],
          padding=1)
    ]
    for i in range(1, len(encoder_conv_filters)):
      self.encoder_layers.append(
          Conv2d(in_channels=encoder_conv_filters[i-1],
                 out_channels=encoder_conv_filters[i],
                 kernel_size=encoder_conv_kernel_size[i],
                 stride=encoder_conv_strides[i],
                 padding=1)    
      )

      self.encoder_layers.append(LeakyReLU())

      if use_batch_norm:
        self.encoder_layers.append(BatchNorm2d(num_features=encoder_conv_filters[i]))
      
      if use_dropout:
        self.encoder_layers.append(Dropout())
    
    # self.fc = Linear(in_features=64 * 7 * 7, out_features=z_dim)

    # Outputting mu and log_var for each point in a latent space

    self.fc1 = Linear(in_features=64 * 7 * 7, out_features=z_dim)
    self.fc2 = Linear(in_features=64 * 7 * 7, out_features=z_dim)

  def forward(self, x):
    for layer in self.encoder_layers:
      x = layer(x)
    
    self.shape_before_flatten = x.shape
    x = x.view(1, -1)
    # x = self.fc(x)
    self.mu = self.fc1(x)
    self.log_var = self.fc2(x)

    def sampling(*args):
      mu, log_var = args
      epsilon = torch.normal(mean=0.0, std=1.0, size=mu.shape)
      return mu + torch.exp(log_var / 2) * epsilon
    
    x = sampling(self.mu, self.log_var)
    return x

  def kl_loss(self, y_true, y_pred):
    """
    Latent loss - Kullback-Leibler Divergence
    """
    return -0.5 * torch.sum(1 + self.log_var - torch.square(self.mu) - torch.exp(self.log_var), dim=1)

  def mse_loss(self, y_true, y_pred):
    """
    Reconstruction/Generative loss - MSE loss
    """
    return torch.mean(torch.square(y_true - y_pred), dim=(1, 2, 3))


class Decoder(nn.Module):
  def __init__(self,
               decoder_conv_t_filters: List[int],
               decoder_conv_t_kernel_size: List[int],
               decoder_conv_t_strides: List[int],
               z_dim: int,
               use_batch_norm: bool = False,
               use_dropout: bool = False):
    super(Decoder, self).__init__()

    self.shape_before_flatten = (-1, 64, 7, 7)
    self.fc = Linear(in_features=z_dim, out_features=np.prod(self.shape_before_flatten[1:]))
    self.decoder_layers = [
      ConvTranspose2d(in_channels=self.shape_before_flatten[1], 
                      out_channels=decoder_conv_t_filters[0], 
                      kernel_size=decoder_conv_t_kernel_size[0],
                      stride=decoder_conv_t_strides[0],
                      padding=1)
    ]

    for i in range(1, len(decoder_conv_t_filters)):
      self.decoder_layers.append(
        ConvTranspose2d(in_channels=decoder_conv_t_filters[i-1], 
                        out_channels=decoder_conv_t_filters[i], 
                        kernel_size=decoder_conv_t_kernel_size[i],
                        stride=decoder_conv_t_strides[i],
                        padding=1,
                        output_padding=1 if decoder_conv_t_strides[i] > 1 else 0)
      )

      if i < len(decoder_conv_t_filters) - 1:
        self.decoder_layers.append(LeakyReLU())
      
    self.decoder_layers.append(ReLU())

  def forward(self, x):
    x = self.fc(x)
    x = x.view(self.shape_before_flatten)
    assert x.ndim == 4
    for layer in self.decoder_layers:
      x = layer(x)

    return x


class AutoEncoder(nn.Module):
  def __init__(self,
               input_dim: Tuple[int],
               encoder_conv_filters: List[int],
               encoder_conv_kernel_size: List[int],
               encoder_conv_strides: List[int],
               decoder_conv_t_filters: List[int],
               decoder_conv_t_kernel_size: List[int],
               decoder_conv_t_strides: List[int],
               z_dim: int,
               use_batch_norm: bool = False,
               use_dropout: bool = False):
    super(AutoEncoder, self).__init__()
    self.encoder = Encoder(input_dim=input_dim,
                           encoder_conv_filters=encoder_conv_filters,
                           encoder_conv_kernel_size=encoder_conv_kernel_size,
                           encoder_conv_strides=encoder_conv_strides,
                           z_dim=z_dim)
    self.decoder = Decoder(decoder_conv_t_filters=decoder_conv_t_filters,
                           decoder_conv_t_kernel_size=decoder_conv_t_kernel_size,
                           decoder_conv_t_strides=decoder_conv_t_strides,
                           z_dim=z_dim)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

  def loss_function(self, mse_loss_factor, y_true, y_pred):
    return mse_loss_factor * self.encoder.mse_loss(y_true, y_pred) + self.encoder.kl_loss(y_true, y_pred)

  def train(self, x_train, epochs, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loss = []

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    for epoch in tqdm(range(epochs)):
      optimizer.zero_grad()
      idx = np.random.choice(range(len(x_train)), batch_size)
      x = x_train[idx]
      output = self.forward(x)
      loss = criterion(output, x)
      train_loss.append(loss)
      loss.backward()
      optimizer.step()

    return train_loss


if __name__ == "__main__":
  autoencoder = AutoEncoder(input_dim=(28, 28, 1),
                            encoder_conv_filters=[32, 64, 64, 64],
                            encoder_conv_kernel_size=[3, 3, 3, 3],
                            encoder_conv_strides=[1, 2, 2, 1],
                            decoder_conv_t_filters=[64, 64, 32, 1],
                            decoder_conv_t_kernel_size=[3, 3, 3, 3],
                            decoder_conv_t_strides=[1, 2, 2, 1],
                            z_dim=2)
