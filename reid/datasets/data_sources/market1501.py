import os.path as osp
import re
from glob import glob

from ..builder import DATA_SOURCES
from .reid_data_source import ReIDDataSource


@DATA_SOURCES.register_module()
class Market1501(ReIDDataSource):
    """Market-1501.

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    DATA_SOURCE = 'Market1501'

    def __init__(self, data_root):
        self.data_root = data_root

        self.train_dir = osp.join(self.data_root, 'bounding_box_train')
        self.query_dir = osp.join(self.data_root, 'query')
        self.gallery_dir = osp.join(self.data_root, 'bounding_box_test')

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        super(Market1501, self).__init__(train, query, gallery)

    def process_dir(self, dir_path):
        img_paths = glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            data.append((img_path, pid, camid))

        return data
