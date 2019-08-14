from argparse import ArgumentParser
import os
from forward_pass import predict


def process():
    parser = ArgumentParser(description="Run predictions from trained model")
    parser.add_argument('network_name', type=str,
                        help="location of network save file")
    parser.add_argument('--number', '-n', type=int, default=10,
                        help="number of images to run predictions on, "
                             "default: 10")
    parser.add_argument('--show', '-s', action='store_true',
                        help="gives detailed view of layer outputs"
                             "default: False")
    parser.add_argument('--gpu', '-g', type=int, default=2,
                        help="index of machine GPU to train with; "
                             "default: 2")
    parser.add_argument('--imagefolder', '-i', type=str, default='../data/prediction_images/',
                        help="relative address of folder where generated images are stored; "
                             "default: ../data/generated_images/")
    parser.add_argument('--labelfolder', '-l', type=str, default='../data/prediction_labels/',
                        help="relative address of folder where labels are stored; "
                             "default: ../data/labels/")
    arguments = parser.parse_args()

    if not os.path.isdir(arguments.network_name):
        raise TypeError("Network storage directory '" + arguments.network_name
                        + "' is not a valid directory. Please define a valid location. "
                        + "See help: -h or --help")

    predict(model_loc=arguments.network_location, n_shown=arguments.number,
            show_layers=arguments.show, gpu_idx=arguments.gpu,
            img_loc=arguments.imagefolder, label_loc=arguments.labelfolder)


if __name__ == "__main__":
    process()

