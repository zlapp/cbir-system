import logging
import sys
import os
from annoy import AnnoyIndex as AnnoyIndexLib

from src.indexes.base_index import BaseIndex


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class AnnoyIndex(BaseIndex):
    """
    Naive implementation of in memory index, only for simple and fast test cases.
    The class does not generate unique document IDs.
    """

    def __init__(self, index_path=None):
        """
        Initialize MemoryIndex

        :param index_field:
        :param index_path:
        """
        self._index_path = index_path
        self._index = self._initialize_index()
        self._count=0


    def _initialize_index(self):
        return AnnoyIndexLib(2048, 'euclidean')  # Length of item vector that will be indexed

    def insert_document(self, document_dict):        
        self._index.add_item(self._count, document_dict['cnn_basic'])
        self._count+=1
        self._index.build(10) 

    def query_index(self, query_dict, similarity, extractor, return_score=False):
        logging.info('Querying index of size' + str(self._count)))
        # get appropriate similarity measure


        # get query features
        query_features = query_dict['features']

        # compare and rank indexed images
        return self._index.get_nns_by_vector(query_features, 10, search_k=-1, include_distances=return_score)

    def persist_index(self):
        if self._index_path is None:
            self._index_path='index.ann'
        self._index.save(self._index_path)  


if __name__ == '__main__':

    mem_index = MemoryIndex('')
    doc_1 = {'doc_name': '1', 'cnn_basic': [1, 2, 3, 4]}
    doc_2 = {'doc_name': '2', 'cnn_basic': [1, 2, 3, 3]}
    doc_3 = {'doc_name': '3', 'cnn_basic': [4, 5, 6, 7]}
    doc_4 = {'doc_name': '4', 'cnn_basic': [7, 8, 9, 0]}
    doc_test = {'doc_name': '5', 'features': [7, 8, 9, 2]}

    mem_index.insert_document(doc_1)
    mem_index.insert_document(doc_2)
    mem_index.insert_document(doc_3)
    mem_index.insert_document(doc_4)

    print(mem_index.query_index(doc_test, 'euclidean', 'cnn_basic'))



