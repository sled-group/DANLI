import os
import time
import json
import lmdb
import torch
import warnings
import numpy as np

from copy import deepcopy
from tqdm import tqdm

from ..utils import data_util


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, partition, args):
        super().__init__()
        self.data_path = None
        self.data_type = None
        self.data_file = None
        self.vis_feat_path = None
        self.vocab_path = None
        self.partition = partition
        self.args = args
        self.test_mode = False

    def load_data(self, json_path):
        '''
        load data
        '''
        # load jsons with pickle and parse them
        if self.data_path is None or self.data_file is None:
            raise TypeError('Have to specify the path and name of data file')
        with open(os.path.join(self.data_path, self.data_file), 'r') as f:
            self.data = json.load(f)
        # load data
        self._length = len(self.data)
        if self.args.fast_epoch:
            self._length = min(16, self._length)
            self.data = self.data[:self._length]
        print('{} dataset size = {}'.format(self.partition, self._length))

    def load_frames(self, game_id, img_index):
        '''
        load image features from the disk
        '''
        if not hasattr(self, 'vis_feats_lmdb'):
            self.vis_feats_lmdb, self.vis_feats = self.load_lmdb(self.vis_feat_path)
        feats_np = []
        t1 = time.time()
        for img_id in img_index:
            feats_bytes = self.vis_feats.get(f'{game_id}/{img_id}'.encode())
            feats_np.append(np.frombuffer(feats_bytes, dtype=np.float32).reshape(512, 7, 7))
        feats_np = np.stack(feats_np)
        # feats_numpy = torch.frombuffer(feats_bytes, dtype=torch.float32).view(-1, 512, 7, 7)
        t3 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # frames = torch.tensor(feats_np)
        # print('oid: %d, load from imdb: %.4f'%(os.getpid(), t3-t1))
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        #     frames = torch.tensor(feats_numpy)
        return feats_np

    def load_lmdb(self, lmdb_path):
        '''
        load lmdb (should be executed in each worker on demand)
        '''
        database = lmdb.open(
            lmdb_path, readonly=True,
            lock=False, readahead=False, meminit=False, max_readers=1)
        cursor = database.begin(write=False)
        return database, cursor

    def __len__(self):
        '''
        return dataset length
        '''
        return self._length

    def __getitem__(self, idx):
        '''
        get item at index idx
        '''
        raise NotImplementedError

    @property
    def id(self):
        return self.data_file[:-5]

    def __del__(self):
        '''
        close the dataset
        '''
        if hasattr(self, 'feats_lmdb'):
            self.feats_lmdb.close()

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self.id)
