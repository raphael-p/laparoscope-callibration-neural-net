from pre_trained_cnns import _data_import
from tensorflow.keras.models import model_from_json, load_model
import json
import sys
import random
import numpy as np
from tensorflow.keras import backend as K


def predict(img_loc="../data/prediction_images/", batch_name='batch_pred', label_loc="../data/prediction_labels/",
            model_loc='../models/vgg_3heads/', n_shown=3, show_layers=True, gpu_idx=2):
    # model import parameters
    weight_loc = model_name+"/"+model_name+"_weights.h5"
    model_loc = model_name+"/"+model_name+"_model.json"

    # data import
    images, foc_labels, princ_labels, rot_labels, trans_labels = _data_import(batch_name, img_loc, label_loc, n_intrinsic)
    # pick random images
    sample = random.sample(range(0, len(images)), n_shown)
    images = images[sample]
    foc_labels = foc_labels[sample]
    princ_label = sprinc_labels[sample]
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
        pred = model.predict(images)

        if show_layers:
            display_layers = ['flatten', 'activation2', 'focal-fc_out',
                              'principal-fc_out', 'rotation-fc_out', 'translation-fc_out']
            inp = model.input
            outputs = [layer.output for layer in model.layers if layer.name in display_layers]
            functor = K.function([inp, K.learning_phase()], outputs)

            for idx in range(n_shown):
                print("\nImage ", idx, "\n"
                      "========")
                img = images[idx:idx + 1]
                layer_outs = functor([img, 1.])

                print("\nFlatten Layer\n"
                      "-------------")
                print(list(layer_outs[0][0]))

                print("\nDense 1028 Layer\n"
                      "----------------")
                print(list(layer_outs[1][0][0:512]))
                print(list(layer_outs[1][0][512:1028]))

                print("\nFocal Output Layer\n"
                      "------------------")
                print(list(layer_outs[2][0]))

                print("\nFocal Ground Truth\n"
                      "------------------")
                print(list(foc_labels[idx]))

                print("\nPrincipal Output Layer\n"
                      "----------------------")
                print(list(layer_outs[3][0]))

                print("\nPrincipal Ground Truth\n"
                      "----------------------")
                print(list(princ_labels[idx]))

                print("\nRotation Output Layer\n"
                      "---------------------")
                print(list(layer_outs[4][0]))

                print("\nRotation Ground Truth\n"
                      "---------------------")
                print(list(rot_labels[idx]))

                print("\nTranslation Output Layer\n"
                      "------------------------")
                print(list(layer_outs[5][0]))

                print("\nTranslation Ground Truth\n"
                      "------------------------")
                print(list(trans_labels[idx]))
                return

    if not show_layers:
        # display
        print("\n\n\n--------------------PREDICTIONS--------------------\n")
        for idx in range(n_shown):
            print("PREDICTED\n",
                  list(pred[0][idx]),
                  list(pred[1][idx]),
                  list(pred[2][idx]))
            print("EXPECTED\n",
                  list(int_labels[idx]),
                  list(rot_labels[idx]),
                  list(trans_labels[idx]), "\n")
        return


if __name__ == '__main__':
    predict(n_shown=10, model_name='res_l2', split_loss=False)
    sys.exit()
