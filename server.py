import logging
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import skimage.io as io

#from src.feature_extractors import hist_ext
from src.parsers import simple_parser
from src.indexes import file_index

#cnn_extractor = hist_ext.HistogramExtractor()

parser = simple_parser.SimpleParser()


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

    data = request.get_json()

    img = data['image'].split(',', 1)[1]
    file_ending = data['file_ending']
    feature = data['feature']
    evaluation = data['evaluation']

    file_name = 'tmp.' + file_ending

    img_data = base64.b64decode(img)
    with open(file_name, 'wb') as f:
        f.write(img_data)

    query = parser.prepare_query(io.imread(file_name))



    return jsonify({'images': ['img1', 'img2']})

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.debug('flask app debug logging enabled')
    app.run(host='0.0.0.0', port=PORT, debug=True)

