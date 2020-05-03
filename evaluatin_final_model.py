from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

def load_dataset():

    (trainX, trainY), (testX, testY) = mnist.load_data()

    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY

def prep_pixels(train, test):

    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm

def run_test_harness():

    trainX, trainY, testX, testY = load_dataset()

    trainX, testX = prep_pixels(trainX, testX)

    model = load_model('final_model.h5')

    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))

run_test_harness()