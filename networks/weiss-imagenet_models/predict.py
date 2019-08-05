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
    parser.add_argument('--split', '-s', action='store_true',
                        help="indicates if network has split head" \
                             "default: False")
    arguments = parser.parse_args()

    if not os.path.isdir(arguments.network_name):
        raise TypeError("Network storage directory '" + arguments.network_name
                        + "' is not a valid directory. Please define a valid location. "
                        + "See help: -h or --help")

    predict(model_name=arguments.network_name, n_shown=arguments.number, split_loss=arguments.split)


if __name__ == "__main__":
    process()

