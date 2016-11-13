# CNN Level Set
CNN Level Set for image segmentation.

### Prerequisite

1. Pascal VOC 2012 dataset, download from: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>
2. Pre-trained models and VGG features; [download from here](https://drive.google.com/open?id=0BzFf_WMmDYN8dUdYZE9iMEZXS0k), then put it in the main project directory

### Setup

1. Install `miniconda`
2. Do `conda env create`
3. Enter the env `source activate cnn-levelset`
4. Install [TensorFlow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html), or for CPU, run `chmod +x tools/setup_tf.sh && ./setup_tf.sh`
5. Run `python experiment_{localizer|segmenter}.py`
