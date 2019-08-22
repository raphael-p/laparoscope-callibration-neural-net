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
    """
    Retrieves batch names from image folder, splits them into training, validation, and evaluation sets.
    :param img_loc: STR, relative address of the image folder
    :param num: INT, number of batches to retrieve
    :param split: FLOAT, fraction of batches that go to evaluation and validation (same value for both)
    :return: list of STR, test batch names;
             list of STR, train batch names;
             list of STR, validation batch names;
             INT, number of testing images;
             INT, number of training images;
             INT, number of validation images;
    """
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
    """
    retrieves images and labels for a single batch
    :param batch_name: STR, name of the batch to retrieve from
    :param image_location: STR, relative address of the image directory
    :param label_location: STR, relative address of the label directory
    :return: numpy arrays, images and labels
    """
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
    if len(labels[0]) == 15:
        labels_princ = None
        labels_rot = labels[:, 3:12].astype(np.float32)
        labels_trans = labels[:, 12:].astype(np.float32)
    elif len(labels[0]) == 17:
        labels_princ = labels[:, 3:5].astype(np.float32)
        labels_rot = labels[:, 5:14].astype(np.float32)
        labels_trans = labels[:, 14:].astype(np.float32)
    else:
        raise ValueError('Invalid label file: ', label_location)

    # image import
    images = []
    root = image_location+batch_name
    for name in tqdm(filenames):
        img = cvtColor(imread(os.path.join(root, name)), COLOR_BGR2RGB)
        images.append(img)
    images = np.asarray(images, dtype=np.uint8)
    return images, labels_foc, labels_princ, labels_rot, labels_trans


def batch_gen(batch_names, image_location, label_location, batch_size, n_epochs):
    """
    data generator for training, evalutation, or validation
    :param batch_names: list of STR, names of batches to retrieve
    :param image_location: STR, relative address of the image directory
    :param label_location: STR, relative address of the label directory
    :param batch_size: INT, number inputs to generate at each yield
    :param n_epochs: INT, number of times to repeat
    :yield: numpy array, a batch of images;
            TUPLE of numpy arrays, a corresponding batch of labels
    """
    n_batches = len(batch_names)
    for _ in range(n_epochs):
        shuffle(batch_names)
        counter = 0
        while counter < n_batches:
            current_batch = batch_names[counter]
            counter += 1
            x, y_foc, y_princ, y_rot, y_trans = _data_import(current_batch, image_location, label_location)
            for idx in range(0, len(x), batch_size):
                if y_princ is not None:
                    yield x[idx:idx+batch_size], (y_foc[idx : idx + batch_size],
                                                  y_princ[idx : idx + batch_size],
                                                  y_rot[idx : idx + batch_size],
                                                  y_trans[idx : idx + batch_size])
                else:
                    yield x[idx:idx + batch_size], (y_foc[idx : idx + batch_size],
                                                    y_rot[idx : idx + batch_size],
                                                    y_trans[idx : idx + batch_size])


def pre_built(network, inputs):
    """
    select and configures a pre-built ImageNet network
    :param network: STR, name of base network: vgg, resnet, or densenet
    :param inputs: keras layer, input layer to feed to model
    :return: keras layer, output of an ImageNet network
    """
    #
    trainable_block_names = []
    trainable_layers = []
    untrainable_layers = []
    if network == "vgg":
        base_net = VGG19(include_top=False, weights='imagenet', pooling='max')
        trainable_block_names = ['block5', 'block4']
        trainable_layers = ['global_max_pooling2d']
        untrainable_layers = []
    elif network == "resnet":
        base_net = ResNet50(include_top=False, weights='imagenet', pooling='max')
        #trainable_block_names = ['res5', 'bn5', 'activation_4']
        #trainable_layers = ['add_13', 'add_14', 'add_15', 'global_max_pooling2d']
        #untrainable_layers = ['activation_4']
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
    """
    determines if a layer from the pre-built model should be trained
    :param layer_name: STR, name of a model layer
    :param block_names: STR, name of a model block to be trained
    :param inclusion_layers: list of STR, names of model layers to be trained
    :param exclusion_layers: list of STR, names of model layers to not be trained, as an exception to a trainable block
    :return: BOOL, True if the layer is not to be trained, False otherwise
    """
    if layer_name in exclusion_layers:
        return True
    if layer_name in inclusion_layers:
        return False
    for name in block_names:
        if name in layer_name:
            return False
    return True


def generate_model(img_shape, base_net, principal):
    """
    builds the neural network model
    :param img_shape: tuple of INT, shape of input image data
    :param base_net: STR, name of pre-trained ImageNet model to use as model base: vgg, resnet, or densenet
    :param principal: BOOL, True to include principal block, False to exclude it
    :return: keras model, the model to be trained
    """
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
    dense_focal = Dense(2, name='focal-fc_out')(dense_focal)
    if principal:
        #   principal block
        dense_principal = Dense(512, name='principal-fc1')(dense)
        dense_principal = LeakyReLU(alpha=0.01, name='activation_principal')(dense_principal)
        dense_principal = Dense(2, name='principal-fc_out')(dense_principal)
    #   rotation block
    dense_rotation = Dense(512, name='rotation-fc1')(dense)
    dense_rotation = LeakyReLU(alpha=0.01, name='activation_rotation')(dense_rotation)
    dense_rotation = Dense(9, name='rotation-fc_out')(dense_rotation)
    #   translation block
    dense_translation = Dense(512, name='translation-fc1')(dense)
    dense_translation = LeakyReLU(alpha=0.01, name='activation_trans')(dense_translation)
    dense_translation = Dense(3, name='translation-fc_out')(dense_translation)

    # model compilation
    if principal:
        new_model = Model(inputs=img_input,
                          outputs=[dense_focal, dense_principal, dense_rotation, dense_translation],
                          name='CalibNet_' + base_net)
    else:
        new_model = Model(inputs=img_input,
                          outputs=[dense_focal, dense_rotation, dense_translation],
                          name='CalibNet_' + base_net)
    return new_model


def define_model(output_location, img_shape, base_net, principal):
    """
    decides whether to generate a model or load a pre-existing one. The purpose of this method is to resume training if
    it has been properly stopped (it uses information from evaluation to know at which epoch to resume)
    :param output_location: STR, relative location of model's output directory
    :param img_shape: tuple of INT, shape of input image data
    :param base_net: STR, name of pre-trained ImageNet model to use as model base: vgg, resnet, or densenet
    :param principal: BOOL, True to include principal block, False to exclude it
    :return: keras model, the model to be trained;
             INT, epoch number to start training from
    """
    # model import parameters
    model_name = os.path.split(os.path.dirname(output_location))[-1]
    weight_loc = os.path.join(output_location, model_name +"_weights.h5")
    model_loc = os.path.join(output_location, model_name +"_model.json")
    eval_loc = os.path.join(output_location, model_name +"_eval.csv")
    epoch_num = 0
    # look for existing model
    if not os.path.exists(weight_loc):
        return generate_model(img_shape, base_net, principal), epoch_num
    else:
        if os.path.exists(weight_loc):
            # model import
            with open(model_loc, 'r') as f:
                model_json = f.read()
            new_model = model_from_json(model_json)
            new_model.load_weights(weight_loc)
            with open(eval_loc) as f:
                f = csv.reader(f)
                for line in f:
                    try:
                        epoch_num += int(line[-1])
                    except ValueError:
                        continue
            return new_model, epoch_num
        else:
            raise ValueError('cannot find model structure location at '+ model_loc)


def run_model(network="vgg", n_batch=3, epochs=1, minibatch_size=8, gpu_idx=3, has_principal=False,
              img_loc='../data/generated_images/', label_loc='../data/labels/', output_loc='../models/logs_practice/'):
    """
    main function, this setups the model, imports the data, trains the model, evaluates, and saves the results
    :param network: STR, name of pre-trained ImageNet model to use as model base: vgg, resnet, or densenet
    :param n_batch: INT, number of batches to retrieve
    :param epochs: INT, number of epochs to train model for
    :param minibatch_size: INT, size of input batch for training and validation
    :param gpu_idx: INT, index of GPU to use on machine
    :param has_principal: BOOL, True to include principal point prediction in training, False otherwise
    :param img_loc: STR, relative address of the image directory
    :param label_loc: STR, relative address of the label directory
    :param output_loc: STR, relative location of model's output directory
    """
    with tf.device('/device:GPU:'+str(gpu_idx)):
        # model setup
        image_shape = (1080, 1920, 3)
        model, init_epoch = define_model(output_loc, image_shape, network, has_principal)
        model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001),
                      loss=mse,
                      metrics=[mae, mape])

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
        stop_early = EarlyStopping(patience=5, monitor='loss', min_delta=50)
        checkpoint = ModelCheckpoint(os.path.join(checkpoints_dir+'weights_checkpoint.{epoch:02d}.h5'), monitor='loss',
                                     save_best_only=True, save_weights_only=True)
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
            writer.writerow(['MSE', 'epochs'])
            writer.writerow([metrics[0], epochs_trained])
        model.save_weights(os.path.join(output_loc, output_name + '_weights.h5'))
    return


if __name__ == "__main__":
    run_model(network='vgg', n_batch=3, epochs=1, gpu_idx=3)
    sys.exit()
