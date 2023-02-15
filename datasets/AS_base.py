import os
import sys
import numpy as np
import random
import open3d as o3d
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import h5py


def get_mapping(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict


class SegDataset(Dataset):
    def __init__(self, root=None, train=True):
        super(SegDataset, self).__init__()

        self.train = train

        self.pcd = []
        self.center = []
        self.label = []
        if self.train:
            for filename in ['train1.h5', 'train2.h5', 'train3.h5', 'train4.h5']:
                print(filename)
                with h5py.File(root+'/'+filename,'r') as f:
                    self.pcd.append(np.array(f['pcd']))
                    self.center.append(np.array(f['center']))
                    self.label.append(np.array(f['label']))
        else:
            for filename in ['test1.h5', 'test2.h5']:
                print(filename)
                with h5py.File(root+'/'+filename,'r') as f:
                    self.pcd.append(np.array(f['pcd']))
                    self.center.append(np.array(f['center']))
                    self.label.append(np.array(f['label']))
        self.pcd = np.concatenate(self.pcd, axis=0)
        self.center = np.concatenate(self.center,axis=0)
        self.label = np.concatenate(self.label,axis=0)

    def __len__(self):
        return len(self.pcd)

    def augment(self, pc, center):
        flip = np.random.uniform(0, 1) > 0.5
        if flip:
            pc = (pc - center)
            pc[:, 0] *= -1
            pc += center
        else:
            pc = pc - center
            jittered_data = np.clip(0.01 * np.random.randn(150,2048,3), -1*0.05, 0.05)
            jittered_data += pc
            pc = pc + center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center

        return pc

    def __getitem__(self, index):
        pc = self.pcd[index]
        center_0 = self.center[index][0]
        label = self.label[index]
        if self.train:
            pc = self.augment(pc, center_0)

        return pc.astype(np.float32), label.astype(np.int64)


if __name__ == '__main__':
    datasets = SegDataset(root='/share/datasets/AS_data_base_h5', train=False)

