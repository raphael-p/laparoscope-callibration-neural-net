from pre_trained_cnns import _data_import
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
import json
import sys
import random
import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras import backend as K

# defining global variables
foc_labels = None
princ_labels = None
rot_labels = None
trans_labels = None

def predbatch_gen(batch_names, image_location, label_location):
    """
    generator for prediction batches, does not yield labels, instead passes them to global variables
    :param batch_names: STR, name of the batch to retrieve from
    :param image_location: STR, relative address of the image directory
    :param label_location: STR, relative address of the label directory
    :yield: numpy array, a batch of images
    """
    # calling global variables
    global foc_labels
    global princ_labels
    global rot_labels
    global trans_labels

    # looping over prediction batches
    n_batches = len(batch_names)
    counter = 0
    while counter < n_batches:
        current_batch = batch_names[counter]
        counter += 1
        x, y_foc, y_princ, y_rot, y_trans = _data_import(current_batch, image_location, label_location)

        # storing labels into global variables
        if foc_labels is None:
            foc_labels = y_foc
            if y_princ is not None:
                princ_labels = y_princ
            rot_labels = y_rot
            trans_labels = y_trans
        else:
            foc_labels = np.append(foc_labels, y_foc, axis=0)
            if y_princ is not None:
                princ_labels = np.append(princ_labels, y_princ, axis=0)
            rot_labels = np.append(rot_labels, y_rot, axis=0)
            trans_labels = np.append(trans_labels, y_trans, axis=0)

        # passing images
        for idx in range(0, len(x)):
            yield x[idx:idx + 1]


def predict(img_loc="../data/prediction_images/", label_loc="../data/prediction_labels/",
            model_loc='../models/vgg_3heads/', n_shown=0, gpu_idx=2):
    """
    main function, imports a trained model, runs predictions, displays results
    :param img_loc: STR, relative address of the image directory
    :param label_loc: STR, relative address of the label directory
    :param model_loc: STR, relative address of the model's output directory
    :param n_shown: INT, number of images to predict. '0' means all, will display statistics. If non-zero will display
    the actual output of the images and compare to the labels (limited to a single batch).
    :param gpu_idx: INT, index of GPU to use on machine
    """
    # calling global variables
    global foc_labels
    global princ_labels
    global rot_labels
    global trans_labels

    # model import parameters
    model_name = os.path.split(os.path.dirname(model_loc))[-1]
    weight_loc = os.path.join(model_loc, model_name +"_weights.h5")
    model_loc = os.path.join(model_loc, model_name +"_model.json")
    batch_names = next(os.walk(img_loc))[1]

    if n_shown == 0:
        # initialising generator
        n_img = 0
        for batch in batch_names:
            n_img += len(next(os.walk(os.path.join(img_loc, batch)))[2])
        generator = predbatch_gen(batch_names, img_loc, label_loc)
    else:
        # importing a batch, and retrieving a sample of images (and their labels)
        batch_size = 10
        images, foc_labels, princ_labels, rot_labels, trans_labels, = _data_import(batch_names[0], img_loc, label_loc)
        n_img = len(images)
        sample = random.sample(range(0, len(images)), n_shown)  # pick random images
        images = images[sample]
        foc_labels = foc_labels[sample]
        if princ_labels is not None:
            princ_labels = princ_labels[sample]
        rot_labels = rot_labels[sample]
        trans_labels = trans_labels[sample]


    with tf.device('/device:GPU:' + str(gpu_idx)):
        # model import
        with open(model_loc, 'r') as f:
            model_json = f.read()
        model = model_from_json(model_json)
        print(model.summary())
        model.load_weights(weight_loc)

        # evaluate
        if n_shown == 0:
            pred = model.predict_generator(generator, steps=int(n_img), verbose=1)
        elif n_shown > batch_size:
            idx = np.arange(0, n_img + 1, batch_size)
            if n_img%batch_size !=0:
                idx = np.append(idx, n_img)
            for i in tqdm(range(len(idx) - 1)):
                if i == 0:
                    pred = model.predict(images[idx[i]:idx[i+1]])
                    continue
                pred_temp = model.predict(images[idx[i]:idx[i+1]])
                for j in range(len(pred)):
                    pred[j] = np.append(pred[j] ,pred_temp[j], axis=0)
        else:
            pred = model.predict(images)

    # first display mode: shows predictions statistics
    if n_shown == 0:
        # statistics
        dev_foc = np.sqrt(np.sum((foc_labels - pred[0])**2, axis=0) / (n_img - 1))
        mape_foc = np.mean(np.abs((foc_labels - pred[0]) / foc_labels), axis=0)
        if princ_labels is None:
            mape_rot = np.mean(np.abs((rot_labels - pred[1]) / rot_labels))
            mape_trans = np.mean(np.abs((trans_labels - pred[2]) / trans_labels))
        else:
            dev_princ = np.sqrt(np.sum((princ_labels - pred[1]) ** 2, axis=0) / (n_img - 1))
            mape_princ = np.mean(np.abs((princ_labels - pred[1]) / princ_labels), axis=0)
            mape_rot = np.mean(np.abs((rot_labels - pred[2]) / rot_labels))
            mape_trans = np.mean(np.abs((trans_labels - pred[3]) / trans_labels))

        # display
        print("\n\n\n--------------------------STATISTICS--------------------------\n")
        print("\nFOCAL LENGTH")
        print("Standard Deviation\t\t fx=", dev_foc[0], " fy=", dev_foc[1])
        print("Mean Absolute Percentage Error\t fx=", mape_foc[0], " fy=", mape_foc[1])
        if princ_labels is not None:
            print("\nPRINCIPAL POINT")
            print("Standard Deviation\t\t cx=", dev_princ[0], " cy=", dev_princ[1])
            print("Mean Absolute Percentage Error\tcx=", mape_princ[0], " cy=", mape_princ[1])
        print("\nROTATION MATRIX")
        print("Mean Absolute Percentage Error\t", mape_rot)
        print("\nTRANSLATION")
        print("Mean Absolute Percentage Error\t", mape_trans)

    # second display mode: shows outputs and labels
    else:
        print("\n\n\n--------------------------PREDICTIONS--------------------------\n")
        for idx in range(n_shown):
            if princ_labels is None:
                print("PREDICTED\n",
                      list(pred[0][idx]),
                      list(pred[1][idx]),
                      list(pred[2][idx])
                      )
                print("EXPECTED\n",
                      list(foc_labels[idx]),
                      list(rot_labels[idx]),
                      list(trans_labels[idx]), "\n")
            else:
                print("PREDICTED\n",
                      list(pred[0][idx]),
                      list(pred[1][idx]),
                      list(pred[2][idx]),
                      list(pred[3][idx])
                      )
                print("EXPECTED\n",
                      list(foc_labels[idx]),
                      list(princ_labels[idx]),
                      list(rot_labels[idx]),
                      list(trans_labels[idx]), "\n")
    return


if __name__ == '__main__':
    predict(n_shown=10, model_name='res_l2', split_loss=False)
    sys.exit()
