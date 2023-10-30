import os
import clip
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from metric import Metric

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("cache/ViT-B-32.pt", device=device)

root = r'E:\zts\dataset\hyrank_rgb\train'
img_paths = os.listdir(root)
labels = [int(i.split('.')[0].split('_')[-1]) for i in img_paths]
img_paths = [os.path.join(root, i) for i in img_paths]
label_names = [
    'Dense urban fabric',
    'Mineral extraction sites',
    'Non-irrigated arable land',
    'Fruit trees',
    'Olive Groves',
    'Coniferous Forest',
    'Dense sclerophyllous vegetation',
    'Sparse sclerophyllous vegetation',
    'Sparsely vegetated areas',
    'Rocks & sand',
    'Water',
    'Coastal Water'
]
NUM_CLASSES = len(label_names)
metric = Metric(NUM_CLASSES)
tokens = ['A hyperspectral image of {}'.format(i) for i in label_names]

metric.reset()
train_bar = tqdm(range(len(img_paths)), desc='testing', ascii=True)
for ind in train_bar:
    data, label = img_paths[ind], labels[ind]
    image = preprocess(Image.open(data)).unsqueeze(0).to(device)
    text = clip.tokenize(tokens).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        pred = probs.argmax(axis=1)
        metric.add(pred, np.array([label]))


PA, mPA, Ps, Rs, F1S, KC = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s(), metric.KC()
print('PA={:.3f} mPA={:.3f} KC={:.3f}'.format(PA, mPA, KC))
for c in range(NUM_CLASSES):
    print('class={} P={:.3f} R={:.3f} F1={:.3f}'.format(c, Ps[c], Rs[c], F1S[c]))
print(metric.matrix)
