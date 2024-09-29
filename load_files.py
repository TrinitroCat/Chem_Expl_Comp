""" Load dataset files from xlsx """

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from random import seed as randomSeed
from random import shuffle as randomShuffle


class DataBase:
    """
    base class of data.
    """
    def __init__(self):
        self.samp_name = list()
        self.samp_feat = list()
        self.feat_name = list()
        self.samp_label = None
        self._data_dict = None

    def __getitem__(self, item):
        if self._data_dict is None:
            assert len(self.samp_feat) == len(self.samp_name), \
                "Number of samples, sample features, and sample names does not match."
            if self.samp_label is not None:
                self._data_dict = {__name: {'features': self.samp_feat[i], 'labels': self.samp_label[i]}
                                   for i, __name in enumerate(self.samp_name)}
            else:
                self._data_dict = {__name: {'features': self.samp_feat[i], 'labels': None}
                                   for i, __name in enumerate(self.samp_name)}
        elif (not isinstance(self._data_dict, Dict)) or (len(self._data_dict) != len(self.samp_name)):
            raise RuntimeError('Wrong data structures.')

        return self._data_dict[item]

    def __len__(self):
        return len(self.samp_name)

    def __repr__(self):
        desc = (f'DataBase:\n\tTotal {len(self)} samples\n\tTotal {len(self.feat_name)} features for each sample'
                f'\n\tHas labels: {self.samp_label is not None}')
        return desc

class TrainDataBase(DataBase):
    """

    """
    def __init__(self):
        super().__init__()
        self.is_split = False
        self.train_feat = None
        self.train_label = None
        self.train_names = None
        self.val_feat = None
        self.val_label = None
        self.val_names = None

    def split(self, ratio:float=0.1, force_update:bool=False, seed:int|None=None):
        """
        Split DataBase into training & validation set.
        Args:
            seed:
            force_update:
            ratio: float between (0, 1).
        Returns:
            None
        """
        assert 0. < ratio < 1., 'ratio must be between (0, 1)'
        if not self.is_split or force_update:
            n_val = round(len(self) * ratio)
            n_train = len(self) - n_val
            np.random.seed(seed)
            val_args = np.random.choice(len(self), n_val, replace=False, )
            self.val_feat = self.samp_feat[val_args]
            self.val_names = self.samp_name[val_args]
            self.train_feat = np.delete(self.samp_feat, val_args, axis=0)
            self.train_names = np.delete(self.samp_name, val_args, axis=0)
            if self.samp_label is not None:
                self.val_label = self.samp_label[val_args]
                self.train_label = np.delete(self.samp_label, val_args, axis=0)
            self.is_split = True

    def blocked_split(self,
                      block_indices:List[int],
                      is_block_shuffle:bool=False,
                      block_split_ratio:float=0.1,
                      force_update:bool=False,
                      inner_split_ratio:float=0.,
                      seed:int|None=None):
        """

        Args:
            block_indices: group division indices: (s1, s2, ..., sn) would split dataset into blocks [:s1], [s1:s2], ..., [sn:] ,
                           random shuffled (optional), and then chose the last `block_split_ratio` of these blocks into validation.
            is_block_shuffle: bool, whether shuffled on block level
            block_split_ratio: float between (0, 1). Ratio of validation set on the block level.
            force_update: bool, if True, data set would be split even though it had been split.
            inner_split_ratio: float, re-split within specific training set and append to validation set.
            seed: random seed of split.

        Returns:
            None
        """
        if not self.is_split or force_update:
            # block level
            data_list = np.split(self.samp_feat, block_indices, axis=0)
            name_list = np.split(self.samp_name, block_indices, axis=0)
            if self.samp_label is not None:
                label_list = np.split(self.samp_label, block_indices, axis=0)
            n_block = len(data_list)
            n_val = round(n_block * block_split_ratio)

            if is_block_shuffle:
                randomSeed(seed)
                if self.samp_label is not None:
                    __comb = list(zip(data_list, name_list, label_list))
                    randomShuffle(__comb)
                    data_list, name_list, label_list = zip(*__comb)
                else:
                    __comb = list(zip(data_list, name_list))
                    randomShuffle(__comb)
                    data_list, name_list = zip(*__comb)

            self.train_feat = np.concatenate(data_list[n_val:], axis=0)
            self.train_names = np.concatenate(name_list[n_val:], axis=0)
            self.val_feat = np.concatenate(data_list[:n_val], axis=0)
            self.val_names = np.concatenate(name_list[:n_val], axis=0)
            if self.samp_label is not None:
                self.train_label = np.concatenate(label_list[n_val:], axis=0)
                self.val_label = np.concatenate(label_list[:n_val], axis=0)
            self.is_split = True
            # inner block level
            if abs(inner_split_ratio - 0.) > 1e-7:
                n_val = round(len(self.train_feat) * inner_split_ratio)
                n_train = len(self.train_feat) - n_val
                np.random.seed(seed)
                val_args = np.random.choice(len(self.train_feat), n_val, replace=False, )
                self.val_feat = np.concatenate([self.val_feat, self.train_feat[val_args]], axis=0)
                self.val_names = np.concatenate([self.val_names, self.train_names[val_args]], axis=0)
                self.train_feat = np.delete(self.train_feat, val_args, axis=0)
                self.train_names = np.delete(self.train_names, val_args, axis=0)
                if self.samp_label is not None:
                    self.val_label = np.concatenate([self.val_label, self.train_label[val_args]], axis=0)
                    self.train_label = np.delete(self.train_label, val_args, axis=0)


def load_files(path: str, label_col: int|None=-1, is_remove_nan: bool=True):
    """

    Args:
        is_remove_nan:
        path:
        label_col:

    Returns:

    """
    data = pd.read_excel(path, header=None)
    data = data.values
    db = TrainDataBase()
    if label_col is not None:
        db.feat_name = data[0, 3:label_col].astype(str)
        db.samp_name = data[1:, :3].astype(str)
        db.samp_feat = data[1:, 3:label_col].astype(np.float32)
        db.samp_label = data[1:, label_col:].astype(np.float32)
    else:
        db.feat_name = data[0, 3:].astype(str)
        db.samp_name = data[1:, :3].astype(str)
        db.samp_feat = data[1:, 3:].astype(np.float32)

    if is_remove_nan:
        db.feat_name = db.feat_name[~np.any(np.isnan(db.samp_feat), axis=0)]
        db.samp_feat = db.samp_feat[:, ~np.any(np.isnan(db.samp_feat), axis=0)]

    return db

if __name__ == '__main__':
    data_base = load_files('origin_dataset.xlsx', label_col=-2)
    data_base
    data_base.blocked_split([16, 15, 18], False, 0.5, inner_split_ratio=0.1)
    data_base
