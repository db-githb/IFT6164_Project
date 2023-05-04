import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import precision_score, accuracy_score
from os import path, listdir
from PIL import Image
from tqdm import tqdm