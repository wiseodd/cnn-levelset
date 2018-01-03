# Data documentation

## Using pretrained models

Using these dataset and models below is useful to reproducing the paper's experimental results.

#### Dataset
`data/dataset` contains all images names from VOC2012 that are being used for this project. It is filtered by considering only images that have single object.

#### Labels
`data/labels` contains pickle files that can be loaded with Numpy. These files are in the shape of `N x 20 x 5` where `N` is the number of data, 20 is the number of classes in VOC2012, and 5 is the label plus bounding box, i.e. first column is the class label, and the remaining 4 columns are for bounding box.

#### Models
`data/models` contains the CNN (based on VGG16) pretrained on the above dataset.

## Using custom dataset

To use your own dataset:

1. Populate `data/dataset` and `data/labels` with your own
2. Create new dataset utility class similar to `cnnlevelset/pascalvoc_util.py`.
3. Train the CNN using `experiment_localizer.py`
4. Test the segmentation using `experiment.py`
