import numpy as np
'''
exmple

plot_model = plot_label(pre,'Indian_pines')

temp1=y_.eval(feed_dict={x: data_all})
y_pred=contrary_one_hot(temp1).astype('int32')

img = plot_model.plot_color(y_pred)

color_RGB is from colorbrewer2.org

'''
#os.environ["CUDA_VISIBLE_DEVICES"]='0'


class plot_label(object):
    def __init__(self, data_name: str, num_class: int):
        self.data_name = data_name
        self.dim = num_class+1
        self.color = np.zeros([num_class+1, 3])
        self.set_default()

    def set_default(self):
        if self.data_name == 'Houston':
            self.color[0, :] = [0, 0, 0]
            self.color[1, :] = [141, 211, 199]
            self.color[2, :] = [255, 255, 179]
            self.color[3, :] = [190, 186, 218]
            self.color[4, :] = [251, 128, 114]
            self.color[5, :] = [128, 177, 211]
            self.color[6, :] = [253, 180, 98]
            self.color[7, :] = [179, 222, 105]

        elif self.data_name == 'HyRANK':
            self.color[0, :] = [0, 0, 0]
            self.color[1, :] = [141, 211, 199]
            self.color[2, :] = [255, 255, 179]
            self.color[3, :] = [190, 186, 218]
            self.color[4, :] = [251, 128, 114]
            self.color[5, :] = [128, 177, 211]
            self.color[6, :] = [253, 180, 98]
            self.color[7, :] = [179, 222, 105]
            self.color[8, :] = [252, 205, 229]
            self.color[9, :] = [217, 217, 217]
            self.color[10, :] = [188, 128, 189]
            self.color[11, :] = [204, 128, 189]
            self.color[12, :] = [255, 237, 111]

        elif self.data_name == 'ShangHang':
            self.color[0, :] = [0, 0, 0]
            self.color[1, :] = [141, 211, 199]
            self.color[2, :] = [255, 255, 179]
            self.color[3, :] = [190, 186, 218]

    def change_data(self, pre, img, d):
        c = np.argwhere(pre == d)
        for i in range(len(c)):
            # img[c[i, 0], c[i, 1], :] = self.color[d, :]
            img[c[i, 0], c[i, 1], :] = np.array(self.color[d, :])
        return img

    def plot_color(self, pre):
        # pre1=np.reshape(pre,[self.shape[0],self.shape[1]])
        # print("pre1", pre1.shape)
        h, w = pre.shape
        img = np.zeros([h, w, 3])
        for d in range(self.dim):
            img = self.change_data(pre, img, d)
        return img





