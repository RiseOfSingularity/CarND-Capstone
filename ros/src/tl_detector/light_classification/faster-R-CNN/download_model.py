#based on: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
import os
import six.moves.urllib as urllib
import tarfile

# What model to download.
MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# load pre-train model and weights
if not os.path.exists(MODEL_NAME+'/model.ckpt.index'):
    if not os.path.exists(MODEL_FILE):
        print "Downloading Pre-trained model and weights...(" + DOWNLOAD_BASE+MODEL_FILE+")"
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    print "Extracting Pre-trained model and weights...("+MODEL_FILE+")"
    for file in tar_file.getmembers():
        tar_file.extract(file, os.getcwd())
    print "Done!"
else:
    print "Pre-trained model and weights already downloaded!"

