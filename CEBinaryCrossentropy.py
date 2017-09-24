from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
import keras.callbacks
import numpy as np
import matplotlib.pyplot as plt
import timeit
from keras.models import model_from_json
from sklearn import metrics, preprocessing
import re
#==============================================================================================================================================================
class Params:
    CELEGANS_INPUT_FILE = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\1K_out.txt'
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
class ClassAccuracy(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.class_acc = []
        self.losses = []

    def on_train_end(self, logs={}):
        return self.class_acc

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_true = self.validation_data[1]
        y_pred = self.model.predict(self.validation_data[0])
        y_pred_bin = preprocessing.binarize(y_pred, threshold=0.5)

        correct_classes_total = 0
        total_classes = 0

        for n, sample in enumerate(y_true):
            tmp_true = str(sample)
            tmp_true = tmp_true.replace(' ','')
            tmp_true = tmp_true.replace('[','')
            tmp_true = tmp_true.replace(']','')
            tmp_true = tmp_true.replace('.','')
            file_from_true = tmp_true[:8]
            rank_from_true = tmp_true[8:16]
            file_to_true = tmp_true[16:24]
            rank_to_true = tmp_true[24:32]
            promotion_true = tmp_true[32:36]

            tmp_pred = str(y_pred_bin[n])
            tmp_pred = tmp_pred.replace(' ', '')
            tmp_pred = tmp_pred.replace('[', '')
            tmp_pred = tmp_pred.replace(']', '')
            tmp_pred = tmp_pred.replace('.', '')
            tmp_pred = tmp_pred.replace('\n', '')

            file_from_pred = tmp_pred[:8]
            rank_from_pred = tmp_pred[8:16]
            file_to_pred = tmp_pred[16:24]
            rank_to_pred = tmp_pred[24:32]
            promotion_pred = tmp_pred[32:36]

            ff = False
            rf = False
            ft = False
            ft = False

            if file_from_true == file_from_pred:
                correct_classes_total = correct_classes_total + 1
                ff = True
            if rank_from_true == rank_from_pred:
                correct_classes_total = correct_classes_total + 1
                rf = True
            if file_to_true == file_to_pred:
                correct_classes_total = correct_classes_total + 1
                ft = True
            if rank_to_true == rank_to_pred:
                correct_classes_total = correct_classes_total + 1
                rt = True
            if ff and rf and ft and rt:
                if promotion_true == promotion_pred:
                    correct_classes_total = correct_classes_total + 1
                    total_classes = total_classes + 1
            total_classes = total_classes + 4

        epoch_class_acc = correct_classes_total / total_classes
        self.class_acc.append(epoch_class_acc)
        print("epoch class accuracy: %.2f%%" % (epoch_class_acc * 100))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
#=====================================================================================================================================================
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
def plot_model(model_history, class_acc):
    plt.plot(model_history.history['binary_accuracy'])
    plt.plot(model_history.history['val_binary_accuracy'])
    plt.plot(model_history.history['loss'])
    plt.plot(class_acc.class_acc)
    plt.ylabel('accuracy&loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'loss', 'class_acc'], loc='upper left')
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
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
        print("Serializing model to JSON...")
        model_json = model.to_json()
        with open(Params.CHECKPOINT_FILE_MODEL, 'w') as json_file:
            json_file.write(model_json)
        model.summary()
        checkpoint = ModelCheckpoint(Params.CHECKPOINT_FILE_WEIGHTS, monitor='binary_crossentropy', verbose=2, save_best_only=True, mode='max')
        class_acc = ClassAccuracy()
        callbacks_list = [checkpoint, class_acc]
        start = timeit.default_timer()
        history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size=Params.BATCH_SIZE, epochs=Params.EPOCHS, callbacks=callbacks_list, verbose=1)
        stop = timeit.default_timer()
        score = model.evaluate(testX, testY, verbose=2)
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('Model evaluation on testY, accuracy :' + "%.2f%%" % (score[1] * 100))
        print('Time to train model: ' + str(stop - start))
        if Params.PLOT_MODEL:
            plot_model(history, class_acc)
    else:
        history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size=Params.BATCH_SIZE, epochs=Params.EPOCHS, verbose=2)
        score = model.evaluate(testX, testY, verbose=2)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
        score = model.evaluate(testX, testY, verbose=2)
        if Params.PLOT_MODEL:
            plot_model(history)

else:
    print('Creating model...')
    model = Sequential()
    model.add(Dense(500, input_dim = len(trainX[0]), init = 'RandomNormal', activation = 'relu'))
    model.add(Dense(500, init = 'RandomNormal', activation = 'relu'))
    model.add(Dense(500, init = 'RandomNormal', activation = 'relu'))
    model.add(Dense(500, init = 'RandomNormal', activation = 'relu'))
    #model.add(Dense(500, init = 'normal', activation = 'sigmoid'))
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
        class_acc = ClassAccuracy()
        callbacks_list = [checkpoint, class_acc]
        start = timeit.default_timer()
        history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size = Params.BATCH_SIZE, epochs = Params.EPOCHS, callbacks=callbacks_list, verbose=2)
        stop = timeit.default_timer()
        score = model.evaluate(testX, testY, verbose = 2)
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('Model evaluation on testY, accuracy :' + "%.2f%%" % (score[1] * 100))
        print('Time to train model: '+ str(stop-start))
        if Params.PLOT_MODEL:
            plot_model(history, class_acc)
    else:
        history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size=Params.BATCH_SIZE, epochs=Params.EPOCHS, verbose=2)
        score = model.evaluate(testX, testY, verbose=2)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
        score = model.evaluate(testX, testY, verbose=2)
        if Params.PLOT_MODEL:
            plot_model(history)