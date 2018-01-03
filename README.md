# CNN Level Set
Official source code of our paper:

* Kristiadi, Agustinus, and Pranowo Pranowo. "Deep Convolutional Level Set Method for Image Segmentation." Journal of ICT Research and Applications 11.3 (2017): 284-298. [[pdf](http://journals.itb.ac.id/index.php/jictra/article/download/3887/3046)]

### Setup

1. Install `miniconda`
2. Do `conda env create`
3. Enter the env `source activate cnn-levelset`
4. Install [TensorFlow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html)
  1. For TF installation without GPU, run `chmod +x tools/tf_setup.sh && sh tools/tf_setup.sh`
5. Download Pascal VOC 2012 dataset, from: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>

### Reproducing experimental results

1. Download dataset that is being used for the paper. [Download from here](https://drive.google.com/open?id=0BzFf_WMmDYN8dUdYZE9iMEZXS0k), then unzip it in the main project directory. See `data/README.txt` for documentations of these features
2. Change `cnnlevelset/config.py`
3. Run `python experiment.py`
