
from keras.models import model_from_json

class Params:
    MODEL_PATH = 'C:\\Users\\mbergbauer\\Desktop\ChessElegans\\2017-09-11_11_41_26_730623'
    JSON_FILE = 'model.json'
    WEIGHTS_FILE = 'weights.best.hdf5'


json_file = open(Params.MODEL_PATH + '\\' + Params.JSON_FILE, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(Params.MODEL_PATH + '\\' + Params.WEIGHTS_FILE)