
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

class Params:
    RAW_INPUT = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\500_out.txt'
    TRAIN_PERCENTAGE = 0.9

def read_input_file():
    dataX = []
    dataY = []
    with open(Params.RAW_INPUT) as f:
        for line in f:
            line = line.rstrip()
            line = line.replace('\n', '')
            tmp = line.split('/')
            dataX.append(list(map(int, tmp[0].split(','))))
            dataY.append(list(map(int, tmp[1].split(','))))
    cutoff = int(len(dataX) * Params.TRAIN_PERCENTAGE)
    trainX = np.array(dataX[:cutoff])
    testX = np.array(dataX[cutoff:])
    trainY = np.array(dataY[:cutoff])
    testY = np.array(dataY[cutoff:])

    return trainX, testX, trainY, testY

trainX, testX, trainY, testY = read_input_file()

model = Sequential()
model.add(Dense(4400, input_dim = len(trainX[0]), init = 'normal', activation = 'tanh'))
model.add(Dense(len(trainY[0]),activation = 'tanh'))
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(trainX, trainY, batch_size = 1, epochs = 10)
model.evaluate(testX, testY)