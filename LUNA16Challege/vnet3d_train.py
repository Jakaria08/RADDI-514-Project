import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_vnet3d import Vnet3dModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvimagedata = pd.read_csv('/DATA/jakaria_data/LUNA16Challege/dataprocess/train_X.csv')
    csvmaskdata = pd.read_csv('/DATA/jakaria_data/LUNA16Challege/dataprocess/train_Y.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    Vnet3d = Vnet3dModule(96, 96, 16, channels=1, costname=("dice coefficient",))
    Vnet3d.train(imagedata, maskdata, "Vnet3d.pd", "/DATA/jakaria_data/logs/segmentation/", 0.001, 0.5, 20, 6)


train()
