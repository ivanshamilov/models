from keras.layers import Input, Flatten, Dense, Conv2D, Conv2DTranspose, Dropout, BatchNormalization, LeakyReLU
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam 
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf


class AutoEncoder():

  def __init__(self):
    self.name = "autoencoder"


if __name__ == "__main__":
  ae = AutoEncoder()
