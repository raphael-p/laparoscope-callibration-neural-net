import tensorflow as tf
from tensorflow.keras.applications import VGG19, ResNet50, InceptionV3
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import mae, mape, mse, cosine_similarity
from tensorflow.keras.callbacks import TensorBoard
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
    # split batches into train and test batches
    batch_names = next(os.walk(img_loc))[1]
    file_names = next(os.walk(img_loc))[2]
    if num > len(batch_names):
        raise ValueError("requesting more batches than there are in the data directory")
    n_test = int(num * split)
    test_set = batch_names[:n_test]
    train_set = batch_names[n_test:num]

    # count files in train and test sets
    train_count = 0
    test_count = 0
    for test_batch in test_set:
        test_count += len(next(os.walk(img_loc + test_batch))[2])
    for train_batch in train_set:
        train_count += len(next(os.walk(img_loc + train_batch))[2])
    return test_set, train_set, test_count, train_count


def _data_import(batch_name, image_location, label_location):
    # image import
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
    labels = []
    with open(label_location+batch_name+'.csv', 'rt') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip the heading
        for line in csv_reader:
            labels.append(line)
    labels = np.asarray(labels)
    print("import complete.")
    return images, labels


def batch_gen(batch_names, image_location, label_location, batch_size):
    n_batches = len(batch_names)
    shuffle(batch_names)
    counter = 0
    while counter < n_batches:
        current_batch = batch_names[counter]
        counter += 1
        x_data, y_data = _data_import(current_batch, image_location, label_location)
        for idx in range(0, len(x_data), batch_size):
            yield x_data[idx:idx+batch_size], y_data[idx:idx+batch_size]


def run(network="vgg", n_batch=60, epochs=5, minibatch_size=2,
        img_loc="../data/generated_images/", label_loc="../data/labels/", metrics_file='./logs_practice/', gpu_idx=0):
    with tf.device('/device:GPU:'+str(gpu_idx)):
        # data import
        proportion_of_test_data = 0.1
        test_batch_names, train_batch_names, test_num, train_num = set_split(img_loc, n_batch, proportion_of_test_data)

        # network settings
        img_length = 1920
        img_width = 1080
        channels = 3
        regression_values = 13
        img_shape = (img_width, img_length, channels)

        # select pre-built ImageNet network
        if network == "vgg":
            base_net = VGG19(input_shape=img_shape, include_top=False, weights='imagenet', pooling='max')
        elif network == "resnet":
            base_net = ResNet50(input_shape=img_shape, include_top=False, weights='imagenet', pooling='max')
        elif network == "inception":
            base_net = InceptionV3(input_shape=img_shape, include_top=False, weights='imagenet', pooling='max')
        else:
            raise ValueError("Invalid network name")
        base_net.trainable = False

        # add top part of network
        flattening_layer = Flatten(name='flatten')
        dense_layer_1 = Dense(4096, activation='tanh', name='fc1')
        dense_layer_2 = Dense(4096, activation='tanh', name='fc2')
        prediction_layer = Dense(regression_values, name='predictions')

        # setup model and TensorBoard
        model = Sequential([
            base_net,
            flattening_layer,
            dense_layer_1,
            dense_layer_2,
            prediction_layer
        ])
        tensorboard = TensorBoard(log_dir=metrics_file)
        model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
                      loss=tf.keras.losses.MSE,
                      metrics=[mae, mape, cosine_similarity])
        # plot_model(model, to_file="../data/model.png")
        print(model.summary())

        model.fit_generator(batch_gen(train_batch_names, img_loc, label_loc, minibatch_size), epochs=epochs, verbose=1,
                            steps_per_epoch=int(train_num/minibatch_size), callbacks=[tensorboard])
        model.evaluate_generator(batch_gen(train_batch_names, img_loc, label_loc, minibatch_size), verbose=0,
                                          steps=int(test_num/minibatch_size), callbacks=[tensorboard])

        model.save_weights(metrics_file+list(filter(None, metrics_file.split("/")))[-1]+"_weights.h5")


if __name__ == "__main__":
    run()

