from __future__ import division, print_function, absolute_import
import glob
import os.path as osp

from ..dataset import ImageDataset

dataset_candidate = {
    0: 'SKKU',  # combine all of 1~5 
    1: 'skku_1_5/SKKU', 
    2: 'skku_2_5/SKKU',
    3: 'skku_3_5/SKKU',
    4: 'skku_4_5/SKKU',
    5: 'skku_5_5/SKKU'
}

class SKKU(ImageDataset):
    dataset_dir = dataset_candidate[0]

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'test_probe')
        self.gallery_dir = osp.join(self.dataset_dir, 'test_gallery')

        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)
        train = self.process_dir(self.train_dir)
        super(SKKU, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        data = []

        for img_path in img_paths:
            img_name = osp.splitext(osp.basename(img_path))[0]
            pid, camid = img_name.split('_')
            pid, camid = int(pid), int(camid)
            data.append((img_path, pid, camid))

        return data
