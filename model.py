import csv
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def generator(samples, data_dir, batch_size=32, steering_correction=0):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering_angles = []

            def append_image_with_angle(image, steering_angle):
                # original
                images.append(image)
                steering_angles.append(steering_angle)

                # Augmentation -> flips the image
                images.append(cv2.flip(image, 1))
                steering_angles.append(steering_angle * -1.0)

            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + steering_correction
                steering_right = steering_center - steering_correction

                # read in images from center, left and right cameras
                path = '%sIMG/' % data_dir
                img_center = cv2.imread(path + batch_sample[0].split('/')[-1])
                img_left = cv2.imread(path + batch_sample[1].split('/')[-1])
                img_right = cv2.imread(path + batch_sample[2].split('/')[-1])

                # data generation and augmentation
                append_image_with_angle(img_center, steering_center)
                append_image_with_angle(img_left, steering_left)
                append_image_with_angle(img_right, steering_right)

            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def make_train_validation_generators(data_dir, batch_size, test_size, steering_correction):
    samples = deque()
    with open('%sdriving_log.csv' % data_dir) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=test_size)

    train_generator = generator(train_samples, data_dir=data_dir, batch_size=batch_size,
                                steering_correction=steering_correction)
    validation_generator = generator(validation_samples, data_dir=data_dir, batch_size=batch_size,
                                     steering_correction=steering_correction)

    # length of the dataset is multiplied by 6 because there are 3 images and each images is flipped along its y axis
    return train_generator, validation_generator, len(train_samples) * 6, len(validation_samples) * 6


def make_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    # remove top 50 and bottom 20 lines from the image
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(3, 160, 320)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


def save_training_validation_loss(history_object, diagram_file_name, results_dir):
    """ Save the training and validation loss """
    plt.switch_backend('agg')
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("%s%s" % (results_dir, diagram_file_name))


def train_from_scratch(data_dir, batch_size, test_size, diagram_file_name, results_dir, model_creator,
                       steering_correction, epochs):
    """
    Will train the NNet from zero with provided data. Store a model when is done and output a result
    :param data_dir: directory containing the acquire data
    :param batch_size: how many images should be loaded in a generator call
    :param test_size: percentage split between test and validation data
    :param diagram_file_name: output image diagram file
    :param results_dir: directory where to store results, model and loss diagram
    :param model_creator: function creating returning a NNet to use
    :param steering_correction: angle by which side cameras are shifted
    :param epochs: total number of training epochs
    """

    train_generator, validation_generator, train_data_size, validation_data_size = make_train_validation_generators(
        data_dir=data_dir, batch_size=batch_size, test_size=test_size, steering_correction=steering_correction)

    model = model_creator()

    checkpoint_saver = ModelCheckpoint(
        filepath='%smodel.{epoch:02d}-{val_loss:.3f}.hdf5' % results_dir,
        verbose=1,
        period=1
    )

    history_object = model.fit_generator(
        generator=train_generator,
        samples_per_epoch=train_data_size,
        validation_data=validation_generator,
        nb_val_samples=validation_data_size,
        nb_epoch=epochs,
        verbose=1,
        callbacks=[checkpoint_saver]
    )

    save_training_validation_loss(
        history_object=history_object,
        diagram_file_name=diagram_file_name,
        results_dir=results_dir
    )

    model.save('%smodel.h5' % results_dir)


def main():
    """
    Entry point for the script. Parameters should be altered manually inside here. Look at the function's docstring for
    further infromation.
    """
    train_from_scratch(
        data_dir='./final/',
        batch_size=32,
        test_size=0.2,
        diagram_file_name='loss_diagram.png',
        results_dir='./final/',
        model_creator=make_model,
        steering_correction=0.3,
        epochs=3
    )


def show_summary():
    """ Prints the model structure """
    model = load_model('model.h5')
    model.summary()


if __name__ == '__main__':
    main()
