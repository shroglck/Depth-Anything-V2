import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image
import torch
import numpy as np
import cv2
from depth_anything_v2.dpt import DepthAnythingV2
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as MetricDepthAnythingV2
import matplotlib
from torch.nn import functional as F
import time
from torchvision.transforms import Compose
import pickle
import argparse
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

def params():
    
    parser = argparse.ArgumentParser(description='Check length of dataset')
    parser.add_argument('--data-dir', type=str, default='/data/shresth/octo-data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    params = params()
    
    dataset = tfds.load('fractal20220817_data', data_dir=params.data_dir,
                        split="train")
    depth_dataset = tfds.load('fractal20220817_depth_data',
                              data_dir=params.data_dir, split="train")
    
    reg_data_len = len([i for i in dataset])
    num_images = sum([len(i['steps']) for i in dataset])
    depth_data_len = len([i for i in depth_dataset])
    num_depth_images = sum([len(i['steps']) for i in depth_dataset])
    
    print(f"Regular data length: {reg_data_len}")
    print(f"Number of images: {num_images}")
    print(f"Depth data length: {depth_data_len}")
    print(f"Number of images: {num_depth_images}")