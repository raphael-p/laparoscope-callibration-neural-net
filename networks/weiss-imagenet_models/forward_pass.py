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


def predict(img_loc="../data/prediction_images/", batch_name='batch_pred', label_loc="../data/prediction_labels/",
            model_loc='../models/vgg_3heads/', n_shown=0, gpu_idx=2):
    # model import parameters
    model_name = os.path.split(os.path.dirname(model_loc))[-1]
    weight_loc = os.path.join(model_loc, model_name +"_weights.h5")
    model_loc = os.path.join(model_loc, model_name +"_model.json")

    # data import
    images, foc_labels, princ_labels, rot_labels, trans_labels,  = _data_import(batch_name, img_loc, label_loc)
    n_img = len(images)


    if n_shown != 0:
        # pick random images
        sample = random.sample(range(0, len(images)), n_shown)
        images = images[sample]
        foc_labels = foc_labels[sample]
        if princ_labels:
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
        batch_size = 10
        if n_shown == 0 or n_shown > batch_size:
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

    # display
    if n_shown == 0:
        # statistics
        dev_foc = np.sqrt(np.sum((foc_labels - pred[0])**2, axis=0) / (n_img - 1))
        mape_foc = np.mean(np.abs((foc_labels - pred[0]) / foc_labels))
        if not princ_labels:
            mape_rot = np.mean(np.abs((rot_labels - pred[1]) / rot_labels)) * 100
            mape_trans = np.mean(np.abs((trans_labels - pred[2]) / trans_labels))
        else:
            dev_princ = np.sqrt(np.sum((princ_labels - pred[0]) ** 2, axis=0) / (n_img - 1))
            mape_princ = np.mean(np.abs((princ_labels - pred[0]) / princ_labels))
            mape_rot = np.mean(np.abs((rot_labels - pred[2]) / rot_labels)) * 100
            mape_trans = np.mean(np.abs((trans_labels - pred[3]) / trans_labels))

        print("\n\n\n--------------------------STATISTICS--------------------------\n")
        print("\nFOCAL LENGTH")
        print("Standard Deviation\t\t fx=", dev_foc[0], " fy=", dev_foc[1])
        print("Mean Average Percentage Error\t", mape_foc)
        if princ_labels:
            print("\nPRINCIPAL POINT")
            print("Standard Deviation\t\t fx=", dev_princ[0], " fy=", dev_princ[1])
            print("Mean Average Percentage Error\t", mape_princ)
        print("\nROTATION MATRIX")
        print("Mean Average Percentage Error\t", mape_rot)
        print("\nTRANSLATION")
        print("Mean Average Percentage Error\t", mape_trans)
    else:
        print("\n\n\n--------------------------PREDICTIONS--------------------------\n")
        print(sample)
        for idx in range(n_shown):
            if not princ_labels:
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
