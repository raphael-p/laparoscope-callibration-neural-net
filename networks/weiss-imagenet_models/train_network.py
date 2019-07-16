from argparse import ArgumentParser
import os
from pre_trained_cnns import run


def process():
    parser = ArgumentParser(description="Train calibration model")
    parser.add_argument('--basenet', '-n', type=str, default='vgg',
                        help="base network to use, pre-trained on ImageNet; "
                             "options: vgg, resnet, inception (VGG19, ResNet50, and InceptionV3, respectively); "
                             "default: vgg")
    parser.add_argument('--batch', '-b', type=int, default=60,
                        help="number of batches train with; "
                             "default: 60")
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help="number of epochs to run model for; "
                             "default: 10")
    parser.add_argument('--minibatch', '-m', type=int, default=60,
                        help="size of training mini-batch; "
                             "default: 20")
    parser.add_argument('--imagefolder', '-i', type=str, default='../data/generated_images/',
                        help="relative address of folder where generated images are stored; "
                             "default: ../data/generated_images/")
    parser.add_argument('--labelfolder', '-l', type=str, default='../data/labels/',
                        help="relative address of folder where labels are stored; "
                             "default: ../data/labels/")
    arguments = parser.parse_args()

    if not os.path.isdir(arguments.imagefolder):
        raise TypeError("Generated data storage directory '" + arguments.imagefolder
                        + "' is not a valid directory. Please define a valid location. "
                        + "See help: -h or --help")
    if not os.path.isdir(arguments.labelfolder):
        raise TypeError("Labels directory '" + arguments.labelfolder
                        + "' is not a valid directory. Please define a valid location. "
                        + "See help: -h or --help")

    run(network=arguments.basenet, n_batch=arguments.batch, epochs=arguments.epochs, minibatch_size=arguments.minibatch,
        img_loc=arguments.imagefolder, label_loc=arguments.labelfolder)


if __name__ == "__main__":
    process()

