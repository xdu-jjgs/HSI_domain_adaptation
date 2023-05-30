import os
import scipy.io as sio
import numpy as np
from PIL import Image


def main():
    # Houston
    root = 'E:/zzy/GAN/data/Houston'
    outputpath = 'fig/houston/result_map'
    source_path = os.path.join(root, 'Houston13.mat')
    target_path = os.path.join(root, 'Houston18.mat')
    source_data = sio.loadmat(source_path)['ori_data'].astype('float32')
    target_data = sio.loadmat(target_path)['ori_data'].astype('float32')
    vis_s = source_data[:, :, (46, 26, 10)]
    vis_s = (vis_s - np.min(vis_s))/(np.max(vis_s) - np.min(vis_s))
    vis_t = target_data[:, :, (46, 26, 10)]
    vis_t = (vis_t - np.min(vis_t)) / (np.max(vis_t) - np.min(vis_t))
    ret_im = Image.fromarray(np.uint8(vis_s * 255)).convert('RGB')
    ret_im.save(os.path.join(outputpath, 'source_data.tif'))
    ret_im = Image.fromarray(np.uint8(vis_t * 255)).convert('RGB')
    ret_im.save(os.path.join(outputpath, 'target_data.tif'))
    # HyRANK
    root = 'E:/zzy/GAN/data/HyRANK'
    outputpath = 'fig/hyrank/result_map'
    source_path = os.path.join(root, 'Dioni.mat')
    target_path = os.path.join(root, 'Loukia.mat')
    source_data = sio.loadmat(source_path)['ori_data'].astype('float32')
    target_data = sio.loadmat(target_path)['ori_data'].astype('float32')
    vis_s = source_data[:, :, (78, 35, 10)]
    vis_s = (vis_s - np.min(vis_s)) / (np.max(vis_s) - np.min(vis_s))
    vis_t = target_data[:, :, (78, 35, 10)]
    vis_t = (vis_t - np.min(vis_t)) / (np.max(vis_t) - np.min(vis_t))
    ret_im = Image.fromarray(np.uint8(vis_s * 255)).convert('RGB')
    ret_im.save(os.path.join(outputpath, 'source_data.tif'))
    ret_im = Image.fromarray(np.uint8(vis_t * 255)).convert('RGB')
    ret_im.save(os.path.join(outputpath, 'target_data.tif'))
    # shanghang
    root = 'E:/zzy/GAN/data/ShanghaiHangzhou'
    outputpath = 'fig/shanghang/result_map'
    source_path = os.path.join(root, 'DataCube_ShanghaiHangzhou.mat')
    target_path = os.path.join(root, 'DataCube_ShanghaiHangzhou.mat')
    source_data = sio.loadmat(source_path)['DataCube2'].astype('float32')
    target_data = sio.loadmat(target_path)['DataCube1'].astype('float32')
    vis_s = source_data[:, :, (80, 40, 20)]
    vis_s = (vis_s - np.min(vis_s)) / (np.max(vis_s) - np.min(vis_s))
    vis_t = target_data[:, :, (80, 40, 20)]
    vis_t = (vis_t - np.min(vis_t)) / (np.max(vis_t) - np.min(vis_t))
    ret_im = Image.fromarray(np.uint8(vis_s * 255)).convert('RGB')
    ret_im.save(os.path.join(outputpath, 'source_data.tif'))
    ret_im = Image.fromarray(np.uint8(vis_t * 255)).convert('RGB')
    ret_im.save(os.path.join(outputpath, 'target_data.tif'))
    return None


if __name__ == '__main__':
    main()
