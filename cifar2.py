"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import _pickle

import numpy as np
import resnet

import os


def save_model(model):
    # load tensorflow and keras backend
    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    import keras.backend.tensorflow_backend as K

    ksess = K.get_session()
    print(ksess)

    # transform keras model to tensorflow graph
    # the output will be json-like format
    K.set_learning_phase(0)
    graph = ksess.graph
    kgraph = graph.as_graph_def()
    print(kgraph)

    # define output node
    num_output = 1
    prefix = "output"
    pred = [None]*num_output
    outputName = [None]*num_output
    for i in range(num_output):
        outputName[i] = prefix + str(i)
        pred[i] = tf.identity(model.get_output_at(i), name=outputName[i])
    print('output name: ', outputName)

    # convert variables in the model graph to constants
    constant_graph = graph_util.convert_variables_to_constants(ksess, ksess.graph.as_graph_def(), outputName)

    # save the model in .pb and .txt
    output_dir = "/home/"
    output_graph_name = "keras2tf.pb"
    output_text_name = "keras2tf.txt"
    graph_io.write_graph(constant_graph, output_dir, output_graph_name, as_text=False)
    graph_io.write_graph(constant_graph, output_dir, output_text_name, as_text=True)
    print('saved graph .pb at: {0}\nsaved graph .txt at: {1}'.format(
            os.path.join(output_dir, output_graph_name),
            os.path.join(output_dir, output_text_name)))


PATH = 'C:\\pleiades\\workspace1\\TensorFlowResNet\\'
def unpickle(file):
    fo = open(file, 'rb')
    dict = _pickle.load(fo, encoding='latin-1')
    fo.close()
    return dict

def one_hot_vec(label):
    vec = np.zeros(10)
    vec[label] = 1
    return vec

def load_data():
    x_all = []
    y_all = []
    for i in range (5):
        d = unpickle(PATH + "cifar-10-batches-py\\data_batch_" + str(i+1))
        x_ = d['data']
        y_ = d['labels']
        x_all.append(x_)
        y_all.append(y_)

    d = unpickle(PATH + 'cifar-10-batches-py\\test_batch')
    x_all.append(d['data'])
    y_all.append(d['labels'])

    x = np.concatenate(x_all) / np.float32(255)
    y = np.concatenate(y_all)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))

    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean

    y = map(one_hot_vec, y)
    y = list(y)
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return (X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('resnet18_cifar10.csv')

    batch_size = 32
    nb_classes = 10
    nb_epoch = 10
    data_augmentation = False

    # input image dimensions
    img_rows, img_cols = 32, 32
    # The CIFAR10 images are RGB.
    img_channels = 3

    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_test /= 128.

    #X_train, Y_train, X_test, Y_test = load_data()

    model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)


    #save_model(model)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])



    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper, csv_logger])
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            validation_data=(X_test, Y_test),
                            epochs=nb_epoch, verbose=1, max_q_size=100,
                            callbacks=[lr_reducer, early_stopper, csv_logger])


# evaluate test data
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

save_model(model)
