from datas.base import Dataset
import torch


class DynamicDataset(Dataset):
    def __init__(self):
        super(DynamicDataset, self).__init__()
        self.data = []
        self.gt = []
        self.confid = []
        self.data = torch.tensor(self.data)
        self.gt = torch.tensor(self.gt)
        self.confid = torch.tensor(self.confid)

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.gt[index]

    def __len__(self):
        return len(self.data)

    def reshape(self):
        self.gt = self.gt.numpy().reshape(-1, )

    def get_labels(self):
        return self.gt

    def get_confid(self):
        return self.confid

    def append(self, data, gt, confid):
        self.data = torch.cat((self.data, data), dim=0)
        self.gt = torch.cat((self.gt, gt), dim=0)
        self.confid = torch.cat((self.confid, confid), dim=0)

    def flush(self):
        self.data = []
        self.gt = []
        self.confid = []
        self.data = torch.tensor(self.data)
        self.gt = torch.tensor(self.gt)
        self.confid = torch.tensor(self.confid)

    def print_info(self):
        print(self.data.size())
        print(self.gt.size())
        print(self.confid.size())
