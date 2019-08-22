from argparse import ArgumentParser
import os
from pre_trained_cnns import run_model
import sys


def process():
    parser = ArgumentParser(description="Train calibration model")
    parser.add_argument('--basenet', '-n', type=str, default='vgg',
                        help="base network to use, pre-trained on ImageNet; "
                             "options: vgg, resnet, densenet "
                             "(VGG19, ResNet50, DenseNet201 respectively); "
                             "default: vgg")
    parser.add_argument('--batch', '-b', type=int, default=3,
                        help="number of batches train with; "
                             "default: 3")
    parser.add_argument('--epochs', '-e', type=int, default=1,
                        help="number of epochs to run model for; "
                             "default: 1")
    parser.add_argument('--minibatch', '-m', type=int, default=8,
                        help="size of training mini-batch; "
                             "default: 8")
    parser.add_argument('--gpu', '-g', type=int, default=3,
                        help="index of machine GPU to train with; "
                             "default: 3")
    parser.add_argument('--principal', '-p', action='store_true',
                        help="model includes principal point prediction"
                             "default: False")
    parser.add_argument('--imagefolder', '-i', type=str, default='../data/generated_images/',
                        help="relative address of folder where generated images are stored; "
                             "default: ../data/generated_images/")
    parser.add_argument('--labelfolder', '-l', type=str, default='../data/labels/',
                        help="relative address of folder where labels are stored; "
                             "default: ../data/labels/")
    parser.add_argument('--metrics', type=str, default='./logs_practice/',
                        help="relative address of folder where weights and logs are saved; "
                             "default: ./logs_practice/")
    arguments = parser.parse_args()

    if not os.path.isdir(arguments.imagefolder):
        raise TypeError("Generated data storage directory '" + arguments.imagefolder
                        + "' is not a valid directory. Please define a valid location. "
                        + "See help: -h or --help")
    if not os.path.isdir(arguments.labelfolder):
        raise TypeError("Labels directory '" + arguments.labelfolder
                        + "' is not a valid directory. Please define a valid location. "
                        + "See help: -h or --help")
    if not os.path.isdir(arguments.metrics):
        raise TypeError("Metrics directory '" + arguments.metrics
                        + "' is not a valid directory. Please define a valid location. "
                        + "See help: -h or --help")

    run_model(network=arguments.basenet, n_batch=arguments.batch, epochs=arguments.epochs,
              minibatch_size=arguments.minibatch, gpu_idx=arguments.gpu, has_principal=arguments.principal,
              img_loc=arguments.imagefolder, label_loc=arguments.labelfolder, output_loc=arguments.metrics)


if __name__ == "__main__":
    process()
    sys.exit()

