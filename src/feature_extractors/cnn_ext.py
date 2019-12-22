import sys
import logging
import numpy as np

from src.feature_extractors.base_ext import BaseExtractor
from skimage import transform

import torch
from torch import nn
import torchvision


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info("Using torch device "+str(device))


def pil2tensor(image):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.expand_dims(a, 2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))

    return torch.from_numpy(a).unsqueeze(dim=0).to(device)


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
        model = torchvision.models.resnet50(pretrained=True)

        # remove last fully-connected layer
        model = nn.Sequential(*list(model.children())[:-1])

        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()

        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        model = model.double()
        # logging.info(str(model))
        return model

    def _preprocess_image(self, image):
        # reshape input image if necessary
        if image.shape[:2] != self._resize_dims:
            image = transform.resize(
                image, self._resize_dims, preserve_range=True, mode='constant')
        image = pil2tensor(image)
        return image

    def extract(self, image):
        """
        Extracts abstract features from the given image.

        :param image: image from which features should be extracted
        :return: a numpy array with features, dimensionality depends on class settings.
        """
        # preprocess image (reshaping, mean subtraction, etc.)
        logging.info("Processing image")
        image_proc = self._preprocess_image(image)

        # extract features
        logging.info("Extracting features from image of size " +
                     str(image_proc.size())+" of type "+str(image_proc.dtype))
        image_feats = self.model(image_proc.double()).squeeze()
        logging.info("Extracted features of size "+str(image_feats.size()))

        return image_feats


if __name__ == '__main__':
    import os
    import argparse
    import matplotlib.pyplot as plt
    from skimage import io

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True,
                    help='Path to folder containing')
    ap.add_argument('-q', '--queries', required=True,
                    help='Number of example queries')
    args = vars(ap.parse_args())
    data_path = args['dataset']
    query_num = args['queries']

    print('Loading images...')
    for root, dirs, files in os.walk(data_path, topdown=True):
        imgs = np.array([io.imread(os.path.join(data_path, file_name))
                         for file_name in files])

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
