import os
import random
import matplotlib.pyplot as plt

import torch

# load imageset from path
def load_imageset(path, transform, max_cnt=None, shuffle=False):
    path += '/'
    file_list = os.listdir(path)
    if shuffle:
        random.shuffle(file_list)

    images = []
    metadata = {
        'mean': [0, 0, 0],
        'std': [0, 0, 0],
    }

    for idx, file in enumerate(file_list):
        if max_cnt and idx >= max_cnt:
            break
        img = plt.imread(path + file)

        metadata['mean'] += img.mean(axis=(0, 1))
        metadata['std'] += img.std(axis=(0, 1))

        images.append(transform(img))
    
    images = torch.stack(images)

    metadata['mean'] /= len(images)
    metadata['std'] /= len(images)
    return (images, file_list, metadata)


# Save & Load model