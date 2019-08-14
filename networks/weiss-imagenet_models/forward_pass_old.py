from pre_trained_cnns import _data_import
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json
import sys
import random

from tensorflow.keras import backend as K


def predict(img_loc="../data/prediction_images/", batch_name='batch_pred', label_loc="../data/labels/",
            model_loc='../models/vgg_2heads/', n_shown=10, split_loss=True, gpu_idx=1):

    # model import parameters
    model_name = list(filter(None, model_loc.split("/")))[-1]
    weight_loc = model_loc+model_name+"_weights.h5"
    model_loc = model_loc+model_name+"_model.json"

    # data import
    n_intrinsic = 4
    if split_loss:
        images, int_labels, rot_labels = _data_import(batch_name, img_loc, label_loc, separator=n_intrinsic)
    else:
        images, labels = _data_import(batch_name, img_loc, label_loc)

    with tf.device('/device:GPU:' + str(gpu_idx)):
        # model import
        with open(model_loc, 'r') as f:
            model_json = f.read()
        model = model_from_json(model_json)
        print(model.summary())
        model.load_weights(weight_loc)

        # pick random images
        sample = random.sample(range(0, len(images)), n_shown)
        # evaluate
        pred = model.predict(images[sample])
        if split_loss:
            int_pred, rot_pred = pred
        if pred.shape[1] == n_intrinsic:
            labels = labels[:, 0:n_intrinsic]
            # pick random images

        ##############################################################################
        inp = model.input
        display_layers = ['flatten', 'activation1', 'activation2', 'fc3']
        outputs = [layer.output for layer in model.layers if layer.name in display_layers]
        names = [layer.name for layer in model.layers]
        print(names)
        functor = K.function([inp, K.learning_phase()], outputs)

        img_samp = images[sample]
        for idx in range(n_shown):
            print("\nImage ", idx, "\n"
                  "========")
            img = img_samp[idx:idx + 1]
            # print(img)
            # pred = model.predict(img)
            # print(pred)
            layer_outs = functor([img, 1.])
            print("\nFlatten Layer\n"
                  "-------------")
            print(list(layer_outs[0][0]))
            print("\nDense 2048 Layer\n"
                  "----------------")
            print(list(layer_outs[1][0][0:512]))
            print(list(layer_outs[1][0][512:1028]))
            print(list(layer_outs[1][0][1028:1536]))
            print(list(layer_outs[1][0][1536:2048]))
            #print(list(layer_outs[1][0][2048:2560]))
            #print(list(layer_outs[1][0][2560:3072]))
            #print(list(layer_outs[1][0][3072:3584]))
            #print(list(layer_outs[1][0][3584:4096]))
            print("\nDense 512 Layer\n"
                  "---------------")
            print(list(layer_outs[2][0]))
            print("\nOutput\n"
                  "------")
            print(list(layer_outs[3][0]))
            print("\nGround truth\n"
                  "------------")
            print(list(labels[sample[idx]]))
    sys.exit()
    ##############################################################################

    # display
    print("\n\n\n--------------------PREDICTIONS--------------------\n")
    if split_loss:
        print("\nINTRINSIC\n=========\n")
        print("PREDICTED\t\t\tEXPECTED")
        for idx in range(n_shown):
            print(list(int_pred[idx]), "\t\t", list(int_labels[sample[idx]]))

        print("\n\nROTATION\n========\n")
        for idx in range(n_shown):
            print("PREDICTED\n", list(rot_pred[idx]))
            print("EXPECTED\n", list(rot_labels[sample[idx]]), "\n")
    else:
        for idx in range(n_shown):
            print("PREDICTED\n", list(pred[idx]))
            print("EXPECTED\n", list(labels[sample[idx]]), "\n")

    sys.exit()


if __name__ == '__main__':
    predict(n_shown=10, model_name='res_l2', split_loss=False)
