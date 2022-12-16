import os
import numpy as np
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
from ..build import DATASETS

from .read_bnt import read_bntfile


CLASS_CODES = {
    'ANGER': 0,
    'DISGUST': 1,
    'FEAR': 2,
    'HAPPY': 3,
    'SADNESS': 4,
    'SURPRISE': 5
}
BOSPHORUS_TOTAL_SUBJECT_NUM = 105


def load_data(data_dir, partition, train_subject_num, num_points):
    all_data = []
    all_label = []
    data_dir = Path(data_dir)

    toltol_subject_list = np.arange(BOSPHORUS_TOTAL_SUBJECT_NUM)
    train_subject_list = np.random.choice(
        BOSPHORUS_TOTAL_SUBJECT_NUM,
        train_subject_num,
        replace=False
    )
    test_subject_list = np.setdiff1d(toltol_subject_list, train_subject_list)

    if partition == 'train':
        subject_list = [f'bs{num:0>3d}' for num in train_subject_list]
    else:
        subject_list = [f'bs{num:0>3d}' for num in test_subject_list]

    for subject in subject_list:
        subject_dir = data_dir / subject
        for f in subject_dir.glob("*.bnt"):
            nrows, ncols, data = read_bntfile(f)
            data = data[:, :3]
            # remove all zero points
            data = data[data.sum(1) != 0, :]
            data = data[np.random.choice(data.shape[0], num_points, replace=False), :]
            label_str = f.name.split('_')[2]
            label = CLASS_CODES.get(label_str)
            all_data.append(data)
            all_label.append(label)
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    return all_data, all_label


@DATASETS.register_module()
class Bosphorus(Dataset):
    """
    This is the data loader for Bosphorus Dataset
    num_points: 1024 by default
    data_dir
    train_subject_num: how many subject in train group
    paritition: train or test
    """

    def __init__(self,
                 num_points=1024,
                 data_dir="./data/BosphorusDB",
                 train_subject_num=84,
                 split='train',
                 transform=None
                 ):
        np.random.seed(49)
        self.partition = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.data, self.label = load_data(data_dir, self.partition, train_subject_num, num_points)
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        data = {'pos': pointcloud,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']
        return data

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_classes(self):
        return np.max(self.label) + 1

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """