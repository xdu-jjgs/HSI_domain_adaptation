import torch.utils as utils


class Dataset(utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def name2label(self, name):
        return self.names.index(name)

    def label2name(self, label):
        return self.names[label]

    @property
    def num_classes(self):
        return len(self.labels)

    @property
    def num_channels(self):
        raise NotImplementedError('num_channels() not implemented')

    @property
    def labels(self):
        # e.g. [0, 1, 2]
        raise NotImplementedError('labels() not implemented')

    @property
    def names(self):
        # e.g. ['background', 'road', 'building']
        raise NotImplementedError('names() not implemented')
