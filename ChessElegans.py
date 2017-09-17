from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
#==============================================================================================================================================================
class Params:
    CELEGANS_INPUT_FILE = 'C:\\Users\\michael\\Desktop\\ChessElegans\\5K_out.txt'
    TRAIN_PERCENTAGE = 0.9
    VALIDATION_SPLIT = 0.1
    CHECKPOINT_MODEL = True
    CHECKPOINT_FILE_WEIGHTS = 'C:\\Users\\michael\\Desktop\\ChessElegans\\ModelRun\\weights.best.hdf5'
    CHECKPOINT_FILE_MODEL = 'C:\\Users\\michael\\Desktop\\ChessElegans\\ModelRun\\model.json'
    PLOT_MODEL = True
    BATCH_SIZE = 10
    EPOCHS = 200
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
    plt.plot(model_history.history['categorical_accuracy'])
    plt.plot(model_history.history['val_categorical_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc = 'upper left')
    plt.show()
    #plt.plot(model_history.history['loss'])
    #plt.plot(model_history.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'],loc = 'upper left')
    #plt.show

#==============================================================================================================================================================
np.random.seed(0)
print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
print('Reading file...')
trainX, testX, trainY, testY = read_input_file()
print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
print('Creating model...')
model = Sequential()
model.add(Dense(3000, input_dim = len(trainX[0]), init = 'RandomUniform', activation = 'relu'))
#model.add(Dense(300, init = 'RandomUniform', activation = 'relu'))
#model.add(Dense(100, init = 'normal', activation = 'sigmoid'))
#model.add(Dense(300, init = 'normal', activation = 'sigmoid'))
#model.add(Dense(100, init = 'normal', activation = 'sigmoid'))
model.add(Dense(len(trainY[0]),activation = 'sigmoid'))
print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
print('Compiling model...')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])
if Params.CHECKPOINT_MODEL:
    print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
    print("Serializing model to JSON...")
    model_json = model.to_json()
    with open(Params.CHECKPOINT_FILE_MODEL, 'w') as json_file:
        json_file.write(model_json)
    model.summary()
    checkpoint = ModelCheckpoint(Params.CHECKPOINT_FILE_WEIGHTS, monitor='categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size = Params.BATCH_SIZE, epochs = Params.EPOCHS, callbacks=callbacks_list, verbose=1)
    score = model.evaluate(testX, testY, verbose = 1)
    print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Model evaluation on testY, accuracy :' + "%.2f%%" % (score[1] * 100))
    if Params.PLOT_MODEL:
        plot_model(history)
else:
    history = model.fit(trainX, trainY, validation_split=Params.VALIDATION_SPLIT, batch_size=Params.BATCH_SIZE, epochs=Params.EPOCHS, verbose=1)
    score = model.evaluate(testX, testY, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    score = model.evaluate(testX, testY, verbose=0)
    if Params.PLOT_MODEL:
        plot_model(history)