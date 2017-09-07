
from keras.models import Sequential
from keras.layers import Dense, Activation

class Params:
    RAW_INPUT = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\6000_out.txt'



def read_input_file():
    dataX = []
    dataY = []
    with open(Params.RAW_INPUT) as f:
        for line in f:
            tmp = line.split('/')
            dataX.append(tmp[0].strip().split(','))
            dataY.append(tmp[1].strip().split(','))

    return dataX, dataY


dataX, dataY = read_input_file()
pass