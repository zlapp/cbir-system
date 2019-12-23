

import numpy as np
from .base_parser import BaseParser
from src.feature_extractors.hist_ext import HistogramExtractor
from src.feature_extractors.cnn_ext import CNNExtractor


class V1ImageParser(BaseParser):
    """
    Class prepared documents for storage or query.
    This version will be used in the final report. It computes histograms, conv features, and
    a mix of conv features and histograms.

    Document template:
    {
        'doc_name': String,
        'cnn_basic': Array[Float]
        'hist_basic': Array[Float]
        'cnn_hist': Array[Float]
    }
    """

    def __init__(self):
        self._hist_ext = HistogramExtractor(nbins_per_ch=(9, 3, 3), use_hsv=True, use_regions=True)
        self._cnn_ext = CNNExtractor(output_layers=['block1_conv2', 'block2_conv2', 'block3_conv2',
                                                    'block4_conv2', 'block5_conv2', 'fc2'])

    def _extract_features(self, image):
        # extract features
        cnn_features = self._cnn_ext.extract(image)
        hist_features = self._hist_ext.extract(image)

        # scale vectors
        hist_features /= np.sum(hist_features, keepdims=True)

        return {'cnn_basic': cnn_features.tolist(), 'hist_basic': hist_features.tolist(), 'cnn_hist':cnn_features.tolist()+hist_features.tolist()}

    def prepare_query(self, query_image):
        """
        Prepares image to be used as query, extracts necessary features.

        :param: image that came as a query
        :return: dictionary holding image features (ready to be used as query)
        """
        return self._extract_features(query_image)

    def prepare_document(self, image_name, image):
        """
        Prepares image to be stored in index, extracts necessary features.

        :param: image to be stored in index
        :return: dictionary holding image information (ready to be stored in index)
        """
        extracted_features = self._extract_features(image)
        extracted_features['doc_name'] = image_name
        return extracted_features
