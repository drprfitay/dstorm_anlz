import os
import json
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

from tqdm import tqdm
from multiprocessing.pool import Pool
import traceback

from utils.torch_pointcloud_utils import random_rotate_3d


class DstormDataset(data.Dataset):
    """
    Properties:
        incl_radius - set/unset to include/exclude radius
        data_augmentation - set/unset to augment data - jitter and rotation
    """

    def __init__(self, path):
        self.root = path

        self._incl_radius = True
        self._data_augmentation = False

        self.classes_file = os.path.join(self.root, 'classes.txt')

        self.classes = {}
        with open(self.classes_file, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.classes[ls[1]] = ls[0]

        self.num_classes = len(self.classes)

        df_rows = []
        for label in self.classes:
            dir_point = os.path.join(path, self.classes[label])
            fns = os.listdir(dir_point)

            for fn in fns:
                df_row = {}
                df_row['label'] = int(label)
                df_row['label_class'] = self.classes[label]
                df_row['full_path'] = os.path.abspath(os.path.join(dir_point, fn))

                pointcloud_dict = torch.load(df_row['full_path'])
                df_row['bounding_radius'] = pointcloud_dict['bounding_radius']
                df_row['partition_radius'] = pointcloud_dict['partition_radius']

                df_rows.append(df_row)

        self.df = pd.DataFrame(df_rows)
        self.df_rows = self.df.to_dict(orient='records')
        self.indexes = self.df.index.values
        self.index_to_row = {i: img for (i, img) in zip(self.indexes, self.df_rows)}

        self.len = len(self.df_rows)

    def __getitem__(self, index):

        if not isinstance(index, (int,)):
            index = index.item()

        row = self.index_to_row[index]

        label = row['label']

        pointcloud_dict = torch.load(row['full_path'])
        pointcloud = pointcloud_dict['partition_pointcloud']

        if self.data_augmentation:
            pc = random_rotate_3d(pointcloud)
            # random jitter
            pc += torch.from_numpy(np.random.normal(0, 0.002, size=pc.size())).type(torch.FloatTensor)

        if self.incl_radius:
            result = pointcloud, pointcloud_dict['partition_radius'], label
        else:
            result = pointcloud, label

        return result

    def __len__(self):
        return self.len

    def set_incl_rad(self):
        self.incl_radius = True

    def unset_incl_rad(self):
        self.incl_radius = False

    def set_data_augmentation(self):
        self.incl_radius = True

    def unset_data_augmentation(self):
        self.incl_radius = False

    @property
    def incl_radius(self):
        return self._incl_radius

    @incl_radius.setter
    def incl_radius(self, value=True):
        if not isinstance(value, bool):
            raise ValueError("incl_radius should be 'bool'")
        self._incl_radius = value

    @property
    def data_augmentation(self):
        return self._data_augmentation

    @data_augmentation.setter
    def data_augmentation(self, value=True):
        if not isinstance(value, bool):
            raise ValueError("data_augmentation should be 'bool'")
        self._data_augmentation = value
