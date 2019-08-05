from pre_trained_cnns import _data_import
from tensorflow.keras.models import model_from_json, load_model
import json
import sys


def predict(img_loc="../data/generated_images/", batch_name = 'batch_60', label_loc="../data/labels/",
            model_name='vgg_2heads', n_shown=10, split_loss=True):

    # model import parameters
    weight_loc = model_name+"/"+model_name+"_weights.h5"
    model_loc = model_name+"/"+model_name+"_model.json"

    # data import
    if split_loss:
        n_intrinsic = 4
        images, int_labels, rot_labels = _data_import(batch_name, img_loc, label_loc,
                                                      separator=n_intrinsic, limit=n_shown)
    else:
        images, labels = _data_import(batch_name, img_loc, label_loc, limit=n_shown)

    # model import
    with open(model_loc, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    print(model.summary())
    model.load_weights(weight_loc)

    # evaluate
    pred = model.predict(images)
    if split_loss:
        int_pred, rot_pred = pred

    # display
    print("\n\n\n--------------------PREDICTIONS--------------------\n")
    if split_loss:
        print("\nINTRINSIC\n=========\n")
        print("PREDICTED\t\t\tEXPECTED")
        for idx in range(n_shown):
            print(list(int_pred[idx]), "\t\t", list(int_labels[idx]))

        print("\n\nROTATION\n========\n")
        for idx in range(n_shown):
            print("PREDICTED\n", list(rot_pred[idx]))
            print("EXPECTED\n", list(rot_labels[idx]),"\n")
    else:
        for idx in range(n_shown):
            print("PREDICTED\n", list(pred[idx]))
            print("EXPECTED\n", list(labels[idx]),"\n")

    sys.exit()


if __name__ == '__main__':
    predict(n_shown=10, model_name='res_l2', split_loss=False)

