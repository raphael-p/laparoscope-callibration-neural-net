import tensorflow as tf
from tensorflow.keras.applications import VGG19, ResNet50, InceptionV3
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


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def set_split(img_loc, num, split):
    batch_names = next(os.walk(img_loc))[1]
    if num > len(batch_names):
        raise ValueError("requesting more batches than there are in the data directory")
    shuffle(batch_names)
    n_test = int(num * split)
    test_set = batch_names[:n_test]
    train_set = batch_names[n_test:num]
    return test_set, train_set


def _data_import(batch_name, image_location, label_location):
    # image import
    print("importing images...")
    images = []
    root = image_location+batch_name+'/'
    files = next(os.walk(root))[2]
    for name in tqdm(sorted_nicely(files), desc="importing from "+root):
        try:
            img = cvtColor(imread(root+name), COLOR_BGR2RGB)
            images.append(img)
        except error:
            continue
    images = np.asarray(images, dtype=np.uint8)

    # label import
    print("importing labels...")
    labels = []
    with open(label_location+batch_name+'.csv', 'rt') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip the heading
        for line in csv_reader:
            labels.append(line)
    labels = np.asarray(labels)
    print("import complete.")
    return images, labels


class BatchGenerator:
    # static variables
    image_location = "placeholder"
    label_location = "placeholder"

    def __init__(self, batch_names):
        self.n_batches = len(batch_names)
        self.b_names = batch_names
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.counter < self.n_batches:
            current_batch = self.b_names[self.counter]
            self.counter += 1
            return _data_import(current_batch, BatchGenerator.image_location, BatchGenerator.label_location)
        else:
            raise StopIteration()


def run(network="vgg", n_batch=60, epochs=10, minibatch_size=3,
        img_loc="../data/generated_images/", label_loc="../data/labels/", weight_file='weights/vgg_1.h5', gpu_idx=0):
    with tf.device('/device:GPU:'+str(gpu_idx)):
        # data import
        proportion_of_test_data = 0.2
        BatchGenerator.image_location = img_loc
        BatchGenerator.label_location = label_loc
        test_batch_names, train_batch_names = set_split(img_loc, n_batch, proportion_of_test_data)

        # network settings
        img_length = 1920
        img_width = 1080
        channels = 3
        regression_values = 13
        img_shape = (img_width, img_length, channels)

        if network == "vgg":
            base_net = VGG19(input_shape=img_shape, include_top=False, weights='imagenet', pooling='max')
        elif network == "resnet":
            base_net = ResNet50(input_shape=img_shape, include_top=False, weights='imagenet', pooling='max')
        elif network == "inception":
            base_net = InceptionV3(input_shape=img_shape, include_top=False, weights='imagenet', pooling='max')
        else:
            raise ValueError("Invalid network name")

        base_net.trainable = False

        flattening_layer = Flatten(name='flatten')
        dense_layer_1 = Dense(4096, activation='tanh', name='fc1')
        dense_layer_2 = Dense(4096, activation='tanh', name='fc2')
        prediction_layer = Dense(regression_values, name='predictions')

        model = Sequential([
            base_net,
            flattening_layer,
            dense_layer_1,
            dense_layer_2,
            prediction_layer
        ])

        tensorboard = TensorBoard()

        model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
                      loss=tf.keras.losses.MSE,
                      metrics=[mae, mape, cosine_similarity])

        # plot_model(model, to_file="../data/model.png")
        print(model.summary())

        for e in range(epochs):
            print("\n———EPOCH %d———" % e)
            for X, Y in BatchGenerator(train_batch_names):
                model.fit(X, Y, batch_size=minibatch_size, epochs=1, verbose=1, callbacks=[tensorboard])

        # final evaluation of the model
        for x_test, y_test in BatchGenerator(test_batch_names):
            scores = model.evaluate(x_test, y_test, verbose=0)
            print("MSE (loss): %.2f%%" % (scores[0] * 100))
            print("MAE: %.2f%%" % (scores[1] * 100))
            print("MAPE: %.2f%%" % (scores[2] * 100))
            print("Cosine: %.2f%%" % (scores[3] * 100))

        print("saving weights...")
        model.save_weights(weight_file)

    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(prediction_layer))


if __name__ == "__main__":
    run()

