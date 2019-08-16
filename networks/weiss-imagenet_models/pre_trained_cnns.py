import tensorflow as tf
from tensorflow.keras.applications import VGG19, ResNet50, InceptionV3, DenseNet201
from tensorflow.keras.layers import Dense, Flatten, Input, LeakyReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.losses import mae, mape, mse, cosine_similarity
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
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
        test_count += len(next(os.walk(os.path.join(img_loc, test_batch)))[2])
    for train_batch in train_set:
        train_count += len(next(os.walk(os.path.join(img_loc, train_batch)))[2])
    for valid_batch in valid_set:
        valid_count += len(next(os.walk(os.path.join(img_loc, valid_batch)))[2])
    return test_set, train_set, valid_set, test_count, train_count, valid_count


def _data_import(batch_name, image_location, label_location):
    # label import
    labels = []
    with open(os.path.join(label_location, batch_name+'.csv'), 'rt') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip the heading
        for line in csv_reader:
            labels.append(line)
    labels = np.asarray(labels)
    filenames = labels[:, 0]
    labels_foc = labels[:, 1:3].astype(np.float32)
    labels_rot = labels[:, 3:12].astype(np.float32)
    labels_trans = labels[:, 12:].astype(np.float32)

    # image import
    images = []
    root = image_location+batch_name
    for name in tqdm(filenames):
        img = cvtColor(imread(os.path.join(root, name)), COLOR_BGR2RGB)
        images.append(img)
    images = np.asarray(images, dtype=np.uint8)
    return images, labels_foc, labels_rot, labels_trans


def batch_gen(batch_names, image_location, label_location, batch_size, n_epochs):
    n_batches = len(batch_names)
    for _ in range(n_epochs):
        shuffle(batch_names)
        counter = 0
        while counter < n_batches:
            current_batch = batch_names[counter]
            counter += 1
            x, y_foc, y_rot, y_trans = _data_import(current_batch, image_location, label_location)
            for idx in range(0, len(x), batch_size):
                    yield x[idx:idx+batch_size], (y_foc[idx:idx+batch_size],
                                                  y_rot[idx:idx+batch_size],
                                                  y_trans[idx:idx+batch_size])


def pre_built(network, inputs):
    # select pre-built ImageNet network
    if network == "vgg":
        base_net = VGG19(include_top=False, weights='imagenet', pooling='max')
        trainable_block_names = ['block5', 'block4']
        #trainable_block_names = []
        trainable_layers = ['global_max_pooling2d']
        #trainable_layers = []
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


def _is_untrainable(layer_name, block_names, inclusion_layers, exclusion_layers):
    if layer_name in exclusion_layers:
        return True
    if layer_name in inclusion_layers:
        return False
    for name in block_names:
        if name in layer_name:
            return False
    return True


def generate_model(img_shape, base_net, n_foc, n_rot, n_trans):
    # model definition
    #   common block
    img_input = Input(shape=img_shape, name='inputs')
    pre_trained_model = pre_built(base_net, img_input)
    flattening_layer = Flatten(name='flatten')(pre_trained_model)
    dense = Dense(2048, name='fc1')(flattening_layer)
    dense = LeakyReLU(alpha=0.01, name='activation1')(dense)
    dense = Dense(1028, name='fc2')(dense)
    dense = LeakyReLU(alpha=0.01, name='activation2')(dense)
    #   focal block
    dense_focal = Dense(512, name='focal-fc1')(dense)
    dense_focal = LeakyReLU(alpha=0.01, name='activation_focal')(dense_focal)
    dense_focal = Dense(n_foc, name='focal-fc_out')(dense_focal)
    #   rotation block
    dense_rotation = Dense(512, name='rotation-fc1')(dense)
    dense_rotation = LeakyReLU(alpha=0.01, name='activation_rotation')(dense_rotation)
    dense_rotation = Dense(n_rot, name='rotation-fc_out')(dense_rotation)
    #   translation block
    dense_translation = Dense(512, name='translation-fc1')(dense)
    dense_translation = LeakyReLU(alpha=0.01, name='activation_trans')(dense_translation)
    dense_translation = Dense(n_trans, name='translation-fc_out')(dense_translation)

    # model compilation
    new_model = Model(inputs=img_input, outputs=[dense_focal, dense_rotation, dense_translation],
                  name='CalibNet_' + base_net)
    return new_model


def define_model(output_location, img_shape, base_net, n_foc, n_rot, n_trans):
    # model import parameters
    model_name = os.path.split(os.path.dirname(output_location))[-1]
    weight_loc = os.path.join(output_location, model_name +"_weights.h5")
    model_loc = os.path.join(output_location, model_name +"_model.json")
    eval_loc = os.path.join(output_location, model_name +"_eval.csv")
    epoch_num = 0
    # look for existing model
    if not os.path.exists(weight_loc):
        return generate_model(img_shape, base_net, n_foc, n_rot, n_trans), epoch_num
    else:
        if os.path.exists(weight_loc):
            # model import
            with open(model_loc, 'r') as f:
                model_json = f.read()
            new_model = model_from_json(model_json)
            new_model.load_weights(weight_loc)
            with open(eval_loc) as f:
                f = csv.reader(f)
                next(f)
                for line in f:
                    try:
                        epoch_num += int(line[-1])
                    except ValueError:
                        continue
            return new_model, epoch_num
        else:
            raise ValueError('cannot find model structure location at '+ model_loc)


def run_model(network="vgg", n_batch=60, epochs=5, minibatch_size=2, gpu_idx=3,
        img_loc="../data/generated_images/", label_loc="../data/labels/", output_loc='../models/logs_practice/'):
    # network settings
    img_length = 1920
    img_width = 1080
    channels = 3
    image_shape = (img_width, img_length, channels)
    n_focal = 2
    n_rotation = 9
    n_translation = 3
    with tf.device('/device:GPU:'+str(gpu_idx)):
        # model setup
        model, init_epoch = define_model(output_loc, image_shape, network, n_focal, n_rotation, n_translation)
        model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001),
                      loss=mse,
                      metrics=[mae, mape, cosine_similarity])

        # model visualisation
        output_name = os.path.split(os.path.dirname(output_loc))[-1]
        plot_model(model, to_file=os.path.join(output_loc, output_name + '_map.png'))
        print("\nModel\n"
              "=====")
        print(model.summary())

        # metrics folder setup
        checkpoints_dir = os.path.join(output_loc, 'checkpoints/')
        tb_events_dir = os.path.join(output_loc, 'tb_events/')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        if not os.path.exists(tb_events_dir):
            os.makedirs(tb_events_dir)

        # callbacks
        stop_early = EarlyStopping(patience=5, monitor='loss')
        checkpoint = ModelCheckpoint(os.path.join(checkpoints_dir+'weights_checkpoint.{epoch:02d}.h5'), monitor='loss',
                                     save_best_only=True, save_weights_only=True, save_freq=3)
        tensorboard = TensorBoard(log_dir=tb_events_dir)

        # save model structure
        model_json = model.to_json()
        with open(os.path.join(output_loc, output_name + '_model.json'), 'w') as json_file:
            json_file.write(model_json)

        # data import
        proportion_of_test_data = 0.15
        test_batch_names, train_batch_names, valid_batch_names, test_num, train_num, val_num = set_split(
            img_loc, n_batch, proportion_of_test_data)

        # display
        print("\nDataset\n"
              "=======")
        print('TEST batches (', test_num, 'images ):\n', test_batch_names)
        print('\nTRAIN batches (', train_num, 'images ):\n', train_batch_names)
        print('\nVALIDATION batches (', val_num, 'images ):\n', valid_batch_names, '\n')

        # defining generators
        train_gen = batch_gen(train_batch_names, img_loc, label_loc, minibatch_size, epochs)
        valid_gen = batch_gen(valid_batch_names, img_loc, label_loc, minibatch_size, epochs)
        test_gen = batch_gen(test_batch_names, img_loc, label_loc, 1, 1)

        # model training
        print("\nTraining\n"
              "========")
        r = model.fit_generator(train_gen,
                                validation_data=valid_gen,
                                validation_steps=int(val_num/minibatch_size) - minibatch_size,
                                epochs=epochs, verbose=2, steps_per_epoch=int(train_num/minibatch_size),
                                callbacks=[tensorboard, stop_early, checkpoint], initial_epoch=init_epoch)
        epochs_trained = len(r.history['loss'])

        # model evaluation
        print("\nEvaluation\n"
              "==========")
        metrics = model.evaluate_generator(test_gen, verbose=2, steps=int(test_num) - minibatch_size,
                                           callbacks=[tensorboard])

        # save model weights
        with open(output_loc+output_name+"_eval.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['MSE', 'MAE', 'MAPE', 'cos_sim', 'epochs'])
            writer.writerow([metrics[0], metrics[1], metrics[2], metrics[3], epochs_trained])
        model.save_weights(os.path.join(output_loc, output_name + '_weights.h5'))

        sys.exit()


if __name__ == "__main__":
    run_model(network='vgg', n_batch=3, epochs=1, gpu_idx=3)
