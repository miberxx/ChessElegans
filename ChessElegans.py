import os
import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
#==============================================================================================================================================================
class Params:
    MODEL_RUN_ROOT = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\'
    MODEL_RUN_PATH = ''
    CELEGANS_INPUT_FILE = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\4_out.txt'
    TRAIN_PERCENTAGE = 0.9
    VALIDATION_SPLIT = 0.1
    CHECKPOINT_MODEL = True
    CHECKPOINT_FILE_WEIGHTS = 'weights.best.hdf5'
    CHECKPOINT_PATH_WEIGHTS = ''
    CHECKPOINT_FILE_MODEL = 'model.json'
    CHECKPOINT_PATH_MODEL = ''
    PLOT_MODEL = True


    BATCH_SIZE = 10
    EPOCHS = 5

    def __init__(self):
        dir = str(datetime.datetime.now()).replace(':','_').replace('.','_').replace(' ','_')
        os.makedirs(Params.MODEL_RUN_ROOT + dir)
        Params.MODEL_RUN_PATH = Params.MODEL_RUN_ROOT + dir
        Params.CHECKPOINT_PATH_WEIGHTS = Params.MODEL_RUN_PATH + '\\' + Params.CHECKPOINT_FILE_WEIGHTS
        Params.CHECKPOINT_PATH_MODEL = Params.MODEL_RUN_PATH + '\\' + Params.CHECKPOINT_FILE_MODEL
        pass
#==============================================================================================================================================================
def read_input_file():
    dataX = []
    dataY = []
    with open(Params.CELEGANS_INPUT_FILE) as f:
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
#==============================================================================================================================================================
def plot_model(model_history):
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc = 'upper left')
    plt.show()
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'],loc = 'upper left')
    plt.show

#==============================================================================================================================================================
np.random.seed(0)
Params()

trainX, testX, trainY, testY = read_input_file()

model = Sequential()
model.add(Dense(4400, input_dim = len(trainX[0]), init = 'normal', activation = 'relu'))
model.add(Dense(2200, init = 'normal', activation = 'relu'))
model.add(Dense(1100, init = 'normal', activation = 'relu'))
model.add(Dense(len(trainY[0]),activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
if Params.CHECKPOINT_MODEL:
    checkpoint = ModelCheckpoint(Params.CHECKPOINT_PATH_WEIGHTS, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size = Params.BATCH_SIZE, epochs = Params.EPOCHS, callbacks=callbacks_list, verbose=2)
    score = model.evaluate(testX, testY, verbose = 0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    print("Serializing model to JSON...")
    model_json = model.to_json()
    with open(Params.CHECKPOINT_PATH_MODEL, 'w') as json_file:
        json_file.write(model_json)
    if Params.PLOT_MODEL:
        plot_model(history)
else:
    history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size=Params.BATCH_SIZE, epochs=Params.EPOCHS, verbose=1)
    score = model.evaluate(testX, testY, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    score = model.evaluate(testX, testY, verbose=0)
    if Params.PLOT_MODEL:
        plot_model(history)