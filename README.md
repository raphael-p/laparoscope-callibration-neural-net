# Single image camera calibration with deep neural networks
#### Rapha&euml;l Piccolin
This repository is for training a deep neural network capable of determining the intrinsic and extrinsic parameters
of a laparoscope camera. Additionally, it contains a script for viewing the model output and computing some assessment
metrics.

This is an individual research project for my MSc thesis. It is a part of the SmartLiver project
(https://wp.cs.ucl.ac.uk/mattclarkson/smart-liver-project/) conducted at the Wellcome/EPSRC Centre for
Interventional & Surgical Sciences (WEISS) with the UCL Centre for Medical Image Computing (CMIC),
 at Charles Bell House.

### Dependencies:
See `REQUIREMENTS.txt` for a list of dependencies.

### Data:
- A sample of the data is included in the repository `./data` contains 3 batches + 2 batches for prediction.
This is from the dataset with a variable principal point. 
- `./real_data` contains 33 still frames extracted from WEISS's laparoscope, and the labels. Its labels are
the intrinsic parameters of the laparoscope (extrinsic parameters are not given), and it is the same for each image.
There are three sets of images with different brightness settings (models performed best on `./real_data/images_250`).
- To generate more data, use the code from `https://github.com/raphael-p/weiss-data_generation`.

### Instructions:
- To train a model, run
```bash
python train_network.py
```
This alone is not sufficent, however, there are quite a few options to select. To view these add the `-h` or `--help`
flags to the command.

A certain directory/filename structure is a assumed, but directory locations for images, labels, and models can be specified:
- Images must be stored in a directory containing image batches (separation into batches was necessary during generation, and was kept that way for consistency).
- Each batch must have its labels stored in a csv file of the same name as the batch.
- Models are stored in their own directory within a directory of models. Within that, a model structure is stored in
a file called `<model name>_model.json`, and weights in a file called `<model name>_weights.h5`. These are generated
automatically when training is complete.

- To make predictions using a trained model, run
```bash
python predict.py
```
As before, there are options to be viewed with `-h` or `--help`. The model's storage location must be specified.
The same assumptions about directories and filenames are made. When choosing how many images to predict, choosing '0'
will predict all images in the image directory, and give accuracy statistics for them. Choosing another number will
limit prediction to the first batch in the directory and show the actual outputs, and corresponding labels.

### Testing:
Run tests with:
```bash
python train_network.py -b 3 -e 1 -m 8 -g 1 --basenet vgg --metrics ../models/logs_practice/
```
This runs a network on one epoch, with one batch of images (<400) for each of training, validation, and evaluation. Make
sure the `models/logs_practice/' directory exists before running this command. It is recommended to delete the contents
of the repository after each practice, otherwise the program will attempt to resume from a previous practice.

### Callbacks:
Several keras callbacks are used during training: TensorBoard, EarlyStopping, ModelCheckpoint. TensorBoard is a
visualisation tool by TensorFlow, you can activate it by running:
```bash
tensorboard --logdir=<model location>
```
EarlyStopping will stop training if loss doesn't improve for 3 consecutive epochs. Settings can be changed in
`pre_trained_cnns.py`. ModelCheckpoint regularly saves the best models in terms of epoch loss.

### Resuming Training:
Training is resumed automatically if stopped nicely (the model finishes training and evaluation after a set number of
 epochs). However, if training is interrupted (last epoch weights and epoch number are not saved) a few things need to
 be done:
1. Within the model's directory, move a weight file from the `checkpoints` subdirectory into the main directory, and
rename to `<model name>_weights.h5`.
2. Add a line to `<model name>_eval.csv` (or create the file if it is not already there), with the epoch number you wish
to resume from (you should pick the epoch number on the weights checkpoint file).
