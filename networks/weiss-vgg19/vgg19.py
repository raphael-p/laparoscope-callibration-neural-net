import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import mae, mape, mse, cosine_similarity
from tensorflow.python.keras.callbacks import TensorBoard
from cv2 import cvtColor, imread, COLOR_BGR2RGB, error
import csv
import numpy as np
from tqdm import tqdm
import os
import time
import re
from random import shuffle


def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def data_import(n_batches, image_location, label_location, split=0.2):
    batch_names = next(os.walk(image_location))[1]
    shuffle(batch_names)

    # image import
    print("importing images...")
    images = []
    batch_counter = 0
    for databatch in batch_names:
        if batch_counter >= n_batches:
            break
        root = image_location+databatch+'/'
        files = next(os.walk(root))[2]
        for name in tqdm(sorted_nicely(files), desc="importing from "+root):
            try:
                img = cvtColor(imread(root+name), COLOR_BGR2RGB)
                images.append(img)
            except error:
                continue
        batch_counter += 1
    time.sleep(0.01)  # quick pause to avoid printing overlaps
    print("converting to image list to numpy array...")
    images = np.asarray(images, dtype=np.uint8)

    # label import
    print("importing labels...")
    labels = []
    batch_counter = 0
    for labelbatch in batch_names:
        if batch_counter >= n_batches:
            break
        with open(label_location+labelbatch+'.csv', 'rt') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # skip the heading
            for line in csv_reader:
                labels.append(line)
        batch_counter += 1
    print("converting to label list to numpy array...")
    labels = np.asarray(labels)

    # randomise and split
    print("randomising and splitting data between test and sample sets...")
    return _data_split(images, labels, split=split)


def _data_split(images, labels, split):
    size = images.shape[0]
    perm = np.random.permutation(size)
    images = images[perm]
    labels = labels[perm]
    test_num = int(split * size)

    images_train_ = images[:-test_num]
    labels_train_ = labels[:-test_num]
    images_test_ = images[-test_num:]
    labels_test_ = labels[-test_num:]
    return images_train_, labels_train_, images_test_, labels_test_


if __name__ == "__main__":
    with tf.device('/device:GPU:0'):
        # data import
        n_batch = 1
        proportion_of_test_data = 0.1
        img_loc = "../data/generated_images/"
        label_loc = "../data/labels/"
        X, Y, x_test, y_test = data_import(n_batch, img_loc, label_loc, split=proportion_of_test_data)
        print("train images shape: ", X.shape)
        print("train labels shape: ", Y.shape)
        print("test images shape: ", x_test.shape)
        print("test labels shape: ", y_test.shape)
        print("data import complete.\n")

        # network settings
        img_width = X.shape[2]
        img_length = X.shape[1]
        channels = X.shape[3]
        img_shape = (X.shape[1], X.shape[2], X.shape[3])
        epochs = 1
        minibatch_size = 20

        VGG19_MODEL = VGG19(input_shape=img_shape, include_top=False, weights='imagenet', pooling='avg')
        VGG19_MODEL.trainable = False
        flattening_layer = Flatten(name='flatten')
        dense_layer_1 = Dense(4096, activation='tanh', name='fc1')
        dense_layer_2 = Dense(4096, activation='tanh', name='fc2')
        prediction_layer = Dense(Y.shape[1], name='predictions')

        model = Sequential([
            VGG19_MODEL,
            flattening_layer,
            dense_layer_1,
            dense_layer_2,
            prediction_layer
        ])

        tensorboard = TensorBoard()

        model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
                      loss=tf.keras.losses.MSE,
                      metrics=[mae, mape, mse, cosine_similarity])

        plot_model(model, to_file="../data/model.png")
        print(model.summary())

        model.fit(X, Y, validation_data=(x_test[:20], y_test[:20]), epochs=epochs, batch_size=minibatch_size,
                  verbose=1, callbacks=[tensorboard])

        print("saving weights...")
        model.save_weights('weights/my_model_weights.h5')

    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run())
