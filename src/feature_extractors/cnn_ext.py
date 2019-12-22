import sys
import logging
import numpy as np

from src.feature_extractors.base_ext import BaseExtractor
from skimage import transform

import torch
from torch import nn
import torchvision


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def pil2tensor(image,dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False)).unsqueeze(dim=0)

def image2np(image):
    "Convert from torch style `image` to numpy/matplotlib style."
    res = image.cpu().permute(1,2,0).numpy()
    return res[...,0] if res.shape[2]==1 else res

class CNNExtractor(BaseExtractor):
    """ Class defines a ConvNet based feature extractor.

        The size of the output feature vector depends on the layer
        used as output of the network.
    """

    def __init__(self, output_layers=('fc2',)):
        """
        Initialize ConvNet extractor.

        :param output_layers: Names of layers which should be used as output of feature extractor (string)
        """
        self._resize_dims = (224, 224)
        self._output_layers = output_layers
        self.model = self._initialize_model()

    def _initialize_model(self):
        model = torchvision.models.resnet18(pretrained=True)
        model.eval()
        # remove last fully-connected layer
        new_classifier = nn.Sequential(*list(model.children())[:-1])
        model.classifier = new_classifier
        return model

    def _preprocess_image(self, image):
        # reshape input image if necessary
        if image.shape[:2] != self._resize_dims:
            image = transform.resize(image, self._resize_dims, preserve_range=True, mode='constant')
        image = pil2tensor(image,np.double)
        return image

    def extract(self, image):
        """
        Extracts abstract features from the given image.

        :param image: image from which features should be extracted
        :return: a numpy array with features, dimensionality depends on class settings.
        """
        # preprocess image (reshaping, mean subtraction, etc.)
        image_proc = self._preprocess_image(image)

        # extract features
        logging.info("Extracting features from image of size"+str(image_feats.size()))
        image_feats = self.model(image_proc)
        logging.info("Extracted features of size"+str(image_feats.size()))

        return image_feats


if __name__ == '__main__':
    import os
    import argparse
    import matplotlib.pyplot as plt
    from skimage import io

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True, help='Path to folder containing')
    ap.add_argument('-q', '--queries', required=True, help='Number of example queries')
    args = vars(ap.parse_args())
    data_path = args['dataset']
    query_num = args['queries']

    print('Loading images...')
    for root, dirs, files in os.walk(data_path, topdown=True):
        imgs = np.array([io.imread(os.path.join(data_path, file_name)) for file_name in files])

    print('Extracting features...')
    cnn_ext = CNNExtractor()
    preds = np.array([cnn_ext.extract(img) for img in imgs])

    print('Finding similar images...')
    for _ in range(query_num):
        query_idx = np.random.randint(imgs.shape[0])
        query_img = preds[query_idx]
        sims = np.array([query_img.dot(other.T) for other in preds])
        most_sim = np.argsort(sims)

        plt.subplot(1, 4, 1)
        plt.title('Query')
        plt.imshow(imgs[query_idx])
        plt.subplot(1, 4, 2)
        plt.title('Result #1 (Sim: %.2f)' % sims[most_sim[-2]])
        plt.imshow(imgs[most_sim[-2]])
        plt.subplot(1, 4, 3)
        plt.title('Result #2 (Sim: %.2f)' % sims[most_sim[-3]])
        plt.imshow(imgs[most_sim[-3]])
        plt.subplot(1, 4, 4)
        plt.title('Result #3 (Sim: %.2f)' % sims[most_sim[-4]])
        plt.imshow(imgs[most_sim[-4]])
        plt.show()
