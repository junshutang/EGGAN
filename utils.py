import numpy as np
import string
from PIL import Image
import math
import torch
from tensorboardX import SummaryWriter
from torchvision import transforms as T

def get_id_label(image_name):
    # print("id name" + id_name)
    all_id_list = ['SN001', 'SN002', 'SN003', 'SN004', 'SN005', 'SN006',
                   'SN007', 'SN008', 'SN009', 'SN010', 'SN011', 'SN012',
                   'SN013', 'SN016', 'SN017', 'SN018', 'SN021', 'SN023',
                   'SN024', 'SN025', 'SN026', 'SN027', 'SN028', 'SN029',
                   'SN030', 'SN031', 'SN032']

    index = np.zeros(len(image_name))
    k=0
    for name in image_name:
        name = name.split('_')[0]
        inx = all_id_list.index(name)
        index[k]=inx
        k+=1
    return index

def div_cfd_label(image_name):
    label = image_name.split('-')
    tgtid = label[2]

    return tgtid

def get_random_au(y, au_array, num_columns):
    au = np.zeros((y.shape[0], num_columns))
    for i in range(y.shape[0]):
        rand_sample_id = np.random.randint(0, len(au_array) - 1)
        cond = au_array[rand_sample_id]
        au[i] = cond
    return au

def get_au_array(attr_path):
    lines = [line.rstrip() for line in open(attr_path, 'r')]
    au_array = np.zeros((len(lines), 17))

    # Extract the info from each line
    for idx, line in enumerate(lines):
        split = line.split()
        values = split[1:]
        label = []  # Vector representing the presence of each attribute in each image
        for n in range(len(values)):
            label.append(float(values[n]))
        map(eval, label)
        au_array[idx] = label

    return au_array

def get_au_dict(attr_path):
    lines = [line.rstrip() for line in open(attr_path, 'r')]
    au_dict = {}

    # Extract the info from each line
    for idx, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]
        label = []  # Vector representing the presence of each attribute in each image
        for n in range(len(values)):
            label.append(float(values[n]))
        map(eval, label)
        au_dict[filename] = label

    return au_dict

def load_multi_gpu_to_cpu(model, path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path)
    key = list(pretrained_dict.keys())[0]
    # 1. filter out unnecessary keys
    # 1.1 multi-GPU ->CPU
    if (str(key).startswith('module.')):
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                           k[7:] in model_dict and v.size() == model_dict[k[7:]].size()}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

def load_on_cpu(path):
    # load pth on CPU
    pretrain = torch.load(path, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in pretrain.items():
        if k == 'state_dict':
            state_dict = OrderedDict()
            for keys in v:
                name = keys[7:]  # remove `module.`
                state_dict[name] = v[keys]
            new_state_dict[k] = state_dict
        else:
            new_state_dict[k] = v

    return new_state_dict

def get_transform(image_size=128):

    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform = T.Compose(transform)

    return transform





