from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn import metrics, preprocessing

import numpy as np
import keras.callbacks
import matplotlib.pyplot as plt
import timeit
from keras.models import model_from_json
#==============================================================================================================================================================
class Params:
    CELEGANS_INPUT_FILE = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\0_3K_out.txt'
    TRAIN_PERCENTAGE = 0.9
    VALIDATION_SPLIT = 0.1
    CHECKPOINT_MODEL = True
    CHECKPOINT_FILE_WEIGHTS = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\ModelRun\\weights.best.hdf5'
    CHECKPOINT_FILE_MODEL = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\ModelRun\\model.json'
    LOAD_MODEL = False
    PLOT_MODEL = True
    BATCH_SIZE = 10
    EPOCHS = 25
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
    #plt.plot(model_history.history['categorical_accuracy'])
    #plt.plot(model_history.history['val_categorical_accuracy'])
    #plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['binary_accuracy'])
    plt.plot(model_history.history['val_binary_accuracy'])
    plt.plot(model_history.history['loss'])
    plt.ylabel('accuracy&loss')
    plt.ylabel('accuracy&loss')
    plt.xlabel('epoch')
    plt.legend(['train','test','loss'], loc = 'upper left')
    plt.show()
#==============================================================================================================================================================
def load_model():
    json_file = open(Params.CHECKPOINT_FILE_MODEL, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(Params.CHECKPOINT_FILE_WEIGHTS)
    return loaded_model
#==============================================================================================================================================================
class predictClasses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.skl_acc_score = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_true = self.validation_data[1]
        y_pred = self.model.predict(self.validation_data[0])
        y_pred_binarized = preprocessing.binarize(y_pred, threshold=0.5)
        #acc_score = metrics.accuracy_score(y_true, y_pred_binarized)
        #print('sklearn accuracy score: ' + str(acc_score))
        correct = 0
        for n, sample in enumerate(y_true):
            for i, digit in enumerate(sample):
                if digit == y_pred_binarized[n][i]:
                    correct = +1
                else:
                    continue
        total = len(y_true)*36
        accuracy = correct / total
        print('my acc: ' + str(accuracy))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

np.random.seed(0)
print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
print('Reading file...')
trainX, testX, trainY, testY = read_input_file()
print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
if Params.LOAD_MODEL:
    print('Loading model and weights...')
    model = load_model()
    print('Compiling model...')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    if Params.CHECKPOINT_MODEL:
        print(
            '--------------------------------------------------------------------------------------------------------------------------------------------------------')
        print("Serializing model to JSON...")
        model_json = model.to_json()
        with open(Params.CHECKPOINT_FILE_MODEL, 'w') as json_file:
            json_file.write(model_json)
        model.summary()
        checkpoint = ModelCheckpoint(Params.CHECKPOINT_FILE_WEIGHTS, monitor='acc', verbose=2, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        start = timeit.default_timer()
        history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size=Params.BATCH_SIZE,
                            epochs=Params.EPOCHS, callbacks=callbacks_list, verbose=2)
        stop = timeit.default_timer()
        score = model.evaluate(testX, testY, verbose=0)
        print(
            '--------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('Model evaluation on testY, accuracy :' + "%.2f%%" % (score[1] * 100))
        print('Time to train model: ' + str(stop - start))
        if Params.PLOT_MODEL:
            plot_model(history)
    else:
        history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size=Params.BATCH_SIZE,
                            epochs=Params.EPOCHS, verbose=1)
        score = model.evaluate(testX, testY, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
        score = model.evaluate(testX, testY, verbose=0)
        if Params.PLOT_MODEL:
            plot_model(history)

else:
    print('Creating model...')
    model = Sequential()
    model.add(Dense(400, input_dim = len(trainX[0]), init = 'RandomNormal', activation = 'relu'))
    model.add(Dense(400, init = 'RandomNormal', activation = 'relu'))
    #model.add(Dense(1000, init = 'RandomNormal', activation = 'relu'))
    #model.add(Dense(1000, init = 'RandomNormal', activation = 'relu'))
    #model.add(Dense(100, init = 'normal', activation = 'sigmoid'))
    model.add(Dense(len(trainY[0]),activation = 'sigmoid'))
    print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Compiling model...')
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    if Params.CHECKPOINT_MODEL:
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
        print("Serializing model to JSON...")
        model_json = model.to_json()
        with open(Params.CHECKPOINT_FILE_MODEL, 'w') as json_file:
            json_file.write(model_json)
        model.summary()
        checkpoint = ModelCheckpoint(Params.CHECKPOINT_FILE_WEIGHTS, monitor='binary_accuracy', verbose=2, save_best_only=True, mode='max')
        my_predict_classes = predictClasses()
        callbacks_list = [checkpoint,my_predict_classes]
        start = timeit.default_timer()
        history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size = Params.BATCH_SIZE, epochs = Params.EPOCHS, callbacks=callbacks_list, verbose=2)
        stop = timeit.default_timer()
        score = model.evaluate(testX, testY, verbose = 0)
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('Model evaluation on testY, accuracy :' + "%.2f%%" % (score[1] * 100))
        print('Time to train model: '+ str(stop-start))
        if Params.PLOT_MODEL:
            plot_model(history)
    else:
        history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size=Params.BATCH_SIZE, epochs=Params.EPOCHS, verbose=1)
        score = model.evaluate(testX, testY, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
        score = model.evaluate(testX, testY, verbose=0)
        if Params.PLOT_MODEL:
            plot_model(history)