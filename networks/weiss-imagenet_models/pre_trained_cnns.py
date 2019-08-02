import tensorflow as tf
from tensorflow.keras.applications import VGG19, ResNet50, InceptionV3, DenseNet201
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mae, mape, mse, cosine_similarity
from tensorflow.keras.callbacks import TensorBoard
from cv2 import cvtColor, imread, COLOR_BGR2RGB, error
import csv
import numpy as np
from tqdm import tqdm
import os
import re
from random import shuffle
import sys

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def set_split(img_loc, num, split):
    # split batches into train and test batches
    batch_names = next(os.walk(img_loc))[1]
    if num > len(batch_names):
        raise ValueError("requesting more batches than there are in the data directory")
    shuffle(batch_names)
    n_test = int(num * split)
    if not n_test and num > 2:
        n_test = 1
    test_set = batch_names[:n_test]
    valid_set = batch_names[n_test:2*n_test]
    train_set = batch_names[2*n_test:num]

    # count files in train and test sets
    train_count = 0
    test_count = 0
    valid_count = 0
    for test_batch in test_set:
        test_count += len(next(os.walk(img_loc + test_batch))[2])
    for train_batch in train_set:
        train_count += len(next(os.walk(img_loc + train_batch))[2])
    for valid_batch in valid_set:
        valid_count += len(next(os.walk(img_loc + valid_batch))[2]) 
    return test_set, train_set, valid_set, test_count, train_count, valid_count


def _data_import(batch_name, image_location, label_location, separator):
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
    labels1 = labels[:, 0:separator]
    labels2 = labels[:, separator:]
    print("import complete.")
    return images, labels1, labels2


def batch_gen(batch_names, image_location, label_location, batch_size, n_epochs, separator):
    n_batches = len(batch_names)
    for _ in range(n_epochs):
        shuffle(batch_names)
        counter = 0
        while counter < n_batches:
            current_batch = batch_names[counter]
            counter += 1
            x_data, y_data1, y_data2 = _data_import(current_batch, image_location, label_location, separator)
            for idx in range(0, len(x_data), batch_size):
                yield x_data[idx:idx+batch_size], (y_data1[idx:idx+batch_size], y_data2[idx:idx+batch_size])


def pre_built(network, inputs):
    # select pre-built ImageNet network
    if network == "vgg":
        base_net = VGG19(include_top=False, weights='imagenet', pooling='max')
    elif network == "resnet":
        base_net = ResNet50(include_top=False, weights='imagenet', pooling='max')
    elif network == "inception":
        base_net = InceptionV3(include_top=False, weights='imagenet', pooling='max')
    elif network == "densenet":
        base_net = DenseNet201(include_top=False, weights='imagenet', pooling='max')
    else:
        raise ValueError("Invalid network name")
    base_net.trainable = False
    return base_net(inputs)


def run(network="vgg", n_batch=60, epochs=5, minibatch_size=2, loss="MSE",
        img_loc="../data/generated_images/", label_loc="../data/labels/", output_loc='./logs_practice/', gpu_idx=0):
    #with tf.device('/device:GPU:'+str(gpu_idx)):
        # data import
        proportion_of_test_data = 0.15
        test_batch_names, train_batch_names, valid_batch_names, test_num, train_num, val_num = set_split(
            img_loc, n_batch, proportion_of_test_data)
        output_name = list(filter(None, output_loc.split("/")))[-1]
        print(test_batch_names, test_num)
        print(train_batch_names, train_num)
        print(valid_batch_names, val_num)
        # network settings
        img_length = 1920
        img_width = 1080
        channels = 3
        n_intrinsic = 4
        n_rot = 9
        img_shape = (img_width, img_length, channels)

        # network architecture
        img_input = Input(shape=img_shape, name='inputs')
        pre_trained_layer = pre_built(network, img_input)
        flattening_layer = Flatten(name='flatten')(pre_trained_layer)
        dense = Dense(4096, activation='tanh', name='fc1')(flattening_layer)
        dense = Dense(1024, activation='tanh', name='fc2')(dense)
        dense_intrinsic = Dense(512, activation='tanh', name='fc3-intrinsic')(dense)
        dense_intrinsic = Dense(n_intrinsic, activation='tanh', name='fc4-intrinsic')(dense_intrinsic)
        dense_rotation = Dense(512, activation='tanh', name='fc3-rotation')(dense)
        dense_rotation = Dense(n_rot, activation='tanh', name='fc4-rotation')(dense_rotation)

        model = Model(inputs=img_input, outputs=[dense_intrinsic, dense_rotation])

        tensorboard = TensorBoard(log_dir=output_loc)
        if loss == "MSE":
            loss_fun = mse
            metric_fun_name = "MAE"
            metric_fun = mae
        elif loss == "MAE":
            loss_fun = mae
            metric_fun_name = "MSE"
            metric_fun = mse
        else:
            raise ValueError("loss defined incorrectly, choose mse or mae")
        model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
                      loss=[loss_fun, loss_fun],
                      metrics=[metric_fun, mape, cosine_similarity])
        plot_model(model, to_file=output_loc+output_name+"_model.png")
        print(model.summary())

        # defining generators
        train_gen = batch_gen(train_batch_names, img_loc, label_loc, minibatch_size, epochs, n_intrinsic)
        valid_gen = batch_gen(valid_batch_names, img_loc, label_loc, minibatch_size, epochs, n_intrinsic)
        test_gen = batch_gen(test_batch_names, img_loc, label_loc, 1, 1, n_intrinsic)

        model.fit_generator(train_gen, validation_data=valid_gen, validation_steps=int(val_num/minibatch_size),
                            epochs=epochs, verbose=1,
                            steps_per_epoch=int(train_num/minibatch_size), callbacks=[tensorboard])

        print("Evaluation")
        metrics = model.evaluate_generator(test_gen, verbose=1, steps=int(test_num),
                                           callbacks=[tensorboard])

        with open(output_loc+output_name+"_eval.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([loss, metric_fun_name, "MAPE", "cos_sim"])
            writer.writerow([metrics[0], metrics[1], metrics[2], metrics[3]])
        model.save_weights(output_loc+output_name+"_weights.h5")
        sys.exit()


if __name__ == "__main__":
    run(n_batch=5)
