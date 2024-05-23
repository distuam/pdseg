# pdseg
Plant disease segmentation

mmseg is the configuration file for MMsegmentation. Place the corresponding folder in the corresponding folder of the project directory.https://github.com/open-mmlab/mmsegmentation

mmseg/result contains the experimental results for plant disease segmentation using the SETR model.

deep_learning contains the code for CNN-based models such as FCN, UNET, and DEEPLAB.

deep_learning/segmentation/result contains the experimental results based on CNN models.

*.sh files are example scripts for submitting tasks to SLURM.

other/preprocessing.py is for creating dataset splitting files for binary or multi-class classification.

other/mmseg_preprocessing.py is for creating dataset splitting files for binary or multi-class classification for MMsegmentation.

main funtion in deep_learning/segmentation/run.py
