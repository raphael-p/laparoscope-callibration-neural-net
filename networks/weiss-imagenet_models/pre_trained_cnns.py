import tensorflow as tf
from tensorflow.keras.applications import VGG19, ResNet50, InceptionV3, DenseNet201
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mae, mape, mse, cosine_similarity
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential
from cv2 import cvtColor, imread, COLOR_BGR2RGB, error
import csv
import numpy as np
from tqdm import tqdm
import os
import re
from random import shuffle
import sys


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


def _data_import(batch_name, image_location, label_location, n_int):
    # label import
    labels = []
    with open(label_location+batch_name+'.csv', 'rt') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip the heading
        for line in csv_reader:
            labels.append(line)
    labels = np.asarray(labels)
    filenames = labels[:, 0]
    labels_int = labels[:, 1:n_int+1].astype(np.float32)
    labels_rot = labels[:, n_int+1:n_int+10].astype(np.float32)
    labels_trans = labels[:, n_int+10:].astype(np.float32)
    # image import
    images = []
    root = image_location+batch_name+'/'
    for name in tqdm(filenames):
        img = cvtColor(imread(root+name), COLOR_BGR2RGB)
        images.append(img)
    images = np.asarray(images, dtype=np.uint8)
    #print(filenames[300], labels_int[300], labels_rot[300], labels_trans[300])
    return images, labels_int, labels_rot, labels_trans


def batch_gen(batch_names, image_location, label_location, batch_size, n_epochs, n_int):
    n_batches = len(batch_names)
    for _ in range(n_epochs):
        shuffle(batch_names)
        counter = 0
        while counter < n_batches:
            current_batch = batch_names[counter]
            counter += 1
            x, y_int, y_rot, y_trans = _data_import(current_batch, image_location, label_location, n_int)
            for idx in range(0, len(x), batch_size):
                yield x[idx:idx+batch_size], \
                      (y_int[idx:idx+batch_size], y_rot[idx:idx+batch_size], y_trans[idx:idx+batch_size])


def pre_built(network, inputs):
    # select pre-built ImageNet network
    if network == "vgg":
        base_net = VGG19(include_top=False, weights='imagenet', pooling='max')
        trainable_block_names = ['block5', 'block4']
        trainable_layers = ['global_max_pooling2d']
        untrainable_layers = []
    elif network == "resnet":
        base_net = ResNet50(include_top=False, weights='imagenet', pooling='max')
        trainable_block_names = ['res5', 'bn5', 'activation_4']
        trainable_layers = ['add_13', 'add_14', 'add_15', 'global_max_pooling2d']
        untrainable_layers = ['activation_4']
    elif network == "densenet":
        base_net = DenseNet201(include_top=False, weights='imagenet', pooling='max')
    else:
        raise ValueError("Invalid network name")
    # defining which layers to train
    for layer in base_net.layers:
        if _is_untrainable(layer.name, trainable_block_names, trainable_layers, untrainable_layers):
            layer.trainable = False
    return base_net(inputs)


def metric_names(loss_name):
    if loss_name == "MSE":
        loss_fun = mse
        metric_fun_name = "MAE"
        metric_fun = mae
    elif loss_name == "MAE":
        loss_fun = mae
        metric_fun_name = "MSE"
        metric_fun = mse
    else:
        raise ValueError("loss defined incorrectly, choose mse or mae")
    return loss_fun, metric_fun_name, metric_fun


def run(network="vgg", n_batch=60, epochs=5, minibatch_size=2, loss="MSE", gpu_idx=3,
        img_loc="../data/generated_images/", label_loc="../data/labels/", output_loc='../models/logs_practice/'):
    with tf.device('/device:GPU:'+str(gpu_idx)):
        # data import
        proportion_of_test_data = 0.15
        test_batch_names, train_batch_names, valid_batch_names, test_num, train_num, val_num = set_split(
            img_loc, n_batch, proportion_of_test_data)
        output_name = list(filter(None, output_loc.split("/")))[-1]
        print("\nDataset\n"
              "=======")
        print('TEST batches (', test_num, 'images ):\n', test_batch_names)
        print('\nTRAIN batches (', train_num, 'images ):\n', train_batch_names)
        print('\nVALIDATION batches (', val_num, 'images ):\n', valid_batch_names, '\n')
        # network settings
        img_length = 1920
        img_width = 1080
        channels = 3
        img_shape = (img_width, img_length, channels)
        n_intrinsic = 2
        n_rotation = 9
        n_translation = 3

        # model definition
        #   common block
        img_input = Input(shape=img_shape, name='inputs')
        pre_trained_model = pre_built(network, img_input)
        flattening_layer = Flatten(name='flatten')(pre_trained_model)
        dense = Dense(2048, name='fc1')(flattening_layer)
        dense = LeakyReLU(alpha=0.01, name='activation1')(dense)
        dense = Dense(1028, name='fc2')(dense)
        dense = LeakyReLU(alpha=0.01, name='activation2')(dense)
        #   intrinsic block
        dense_intrinsic = Dense(512, name='int-fc1')(dense)
        dense_intrinsic = LeakyReLU(alpha=0.01, name='activation3')(dense_intrinsic)
        dense_intrinsic = Dense(n_intrinsic, name='int-fc_out')(dense_intrinsic)
        #   rotation block
        dense_rotation = Dense(512, name='rot-fc1')(dense)
        dense_rotation = LeakyReLU(alpha=0.01, name='activation4')(dense_rotation)
        dense_rotation = Dense(n_rotation, name='rot-fc_out')(dense_rotation)
        #   translation block
        dense_translation = Dense(512, name='trans-fc1')(dense)
        dense_translation = LeakyReLU(alpha=0.01, name='activation5')(dense_translation)
        dense_translation = Dense(n_translation, name='trans-fc_out')(dense_translation)

        # model compilation
        model = Model(inputs=img_input, outputs=[dense_intrinsic, dense_rotation, dense_translation], name='CalibNet_'+network)
        loss_fun, metric_fun_name, metric_fun = metric_names(loss)  # setting up which loss functions to use
        model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001),
                      loss=[loss_fun, loss_fun, loss_fun],
                      metrics=[metric_fun, mape, cosine_similarity])

        # model visualisation
        print("\nModel\n"
              "=====")
        tensorboard = TensorBoard(log_dir=output_loc)
        plot_model(model, to_file=output_loc + output_name + "_map.png")
        print(model.summary())
        # save model structure
        model_json = model.to_json()
        with open(output_loc+output_name+"_model.json", "w") as json_file:
            json_file.write(model_json)
        sys.exit()
        # defining generators
        train_gen = batch_gen(train_batch_names, img_loc, label_loc, minibatch_size, epochs, n_intrinsic)
        valid_gen = batch_gen(valid_batch_names, img_loc, label_loc, minibatch_size, epochs, n_intrinsic)
        test_gen = batch_gen(test_batch_names, img_loc, label_loc, 1, 1, n_intrinsic)

        model.fit_generator(train_gen, validation_data=valid_gen, validation_steps=int(val_num/minibatch_size),
                            epochs=epochs, verbose=2, steps_per_epoch=int(train_num/minibatch_size),
                            callbacks=[tensorboard])

        print("Evaluation")
        metrics = model.evaluate_generator(test_gen, verbose=2, steps=int(test_num),
                                           callbacks=[tensorboard])

        # save model weights
        with open(output_loc+output_name+"_eval.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([loss, metric_fun_name, "MAPE", "cos_sim"])
            writer.writerow([metrics[0], metrics[1], metrics[2], metrics[3]])
        model.save_weights(output_loc+output_name+"_weights.h5")

        sys.exit()


if __name__ == "__main__":
    run(network='vgg', n_batch=3, epochs=1, img_loc="../data_cut/generated_images/", label_loc="../data_cut/labels/")
