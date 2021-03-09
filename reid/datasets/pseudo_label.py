from .reid_dataset import ReIDDataset
from .builder import DATASETS


@DATASETS.register_module()
class PseudoLabelDataset(ReIDDataset):

    def update_labels(self, labels):
        assert len(labels) == len(self)

        # update self.img_items
        img_items = []
        for i in range(len(self)):
            img_items.append(
                (self.img_items[i][0], labels[i], self.img_items[i][2]))
        self.img_items = img_items

        # update self.pids
        self.pids = list(set(labels))
        # update self.pid_dict
        self.pid_dict = {p: i for i, p in enumerate(self.pids)}
