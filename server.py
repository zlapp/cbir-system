import logging
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from skimage import io
import os

#from src.feature_extractors import hist_ext
from src.parsers import v1image_parser
from src.indexes import mem_index, es_index

DATA_PATH = 'https://s3.eu-central-1.amazonaws.com/kth-mir/mirflickr/'

parser = v1image_parser.V1ImageParser()

indexer = es_index.ESIndex('elasticsearch:9200')

# indexer = mem_index.MemoryIndex('/app/data/v1_parsed.p1')

PORT = 8081

app = Flask(__name__)
CORS(app)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logging.info('Flask server initialized on port ' + str(PORT))


@app.route('/')
def index():
    ''' Index route '''
    return 'Server is alive!'


@app.route('/search', methods=['POST'])
def search():
    logging.info("Handling search request")

    data = request.get_json()

    img = data['image'].split(',', 1)[1]
    file_ending = data['file_ending']
    feature = data['feature']
    evaluation = data['evaluation']

    file_name = 'tmp.' + file_ending

    img_data = base64.b64decode(img)
    with open(file_name, 'wb') as f:
        f.write(img_data)

    logging.info("Preparing query")
    query = parser.prepare_query(io.imread(file_name))[feature]

    query_dict = {'doc_name': None, 'features': query}

    logging.info("Dispatching query "+str(query_dict))
    query_response = indexer.query_index(
        query_dict, evaluation, feature)[:25]
    logging.info("Completed query" + str(query_response))
    return jsonify({'images': query_response, 'data_path': DATA_PATH})


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.debug('flask app debug logging enabled')
    app.run(host='0.0.0.0', port=PORT, debug=True)

