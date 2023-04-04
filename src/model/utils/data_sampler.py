import torch
import random
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, indices=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # get the indicies and length
        self.indices = [(i, sort_key) for i, sort_key in enumerate(dataset.traj_lengths)]
        # if indices are passed, then use only the ones passed (for ddp)
        if indices is not None:
            self.indices = torch.tensor(self.indices)[indices].tolist()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(self.indices), self.batch_size * 100):
            pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 100], key=lambda x: x[1]))
        self.pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        batches = [self.pooled_indices[i:i + self.batch_size] for i in
                   range(0, len(self.pooled_indices), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.indices) - 1) // self.batch_size + 1


class DistributedBucketBatchSampler(DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas=None, shuffle=True):
        super().__init__(dataset=dataset, num_replicas=num_replicas, shuffle=shuffle)
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        batch_sampler = BucketBatchSampler(self.dataset,
                                           batch_size=self.batch_size,
                                           indices=indices)
        return iter(batch_sampler)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
