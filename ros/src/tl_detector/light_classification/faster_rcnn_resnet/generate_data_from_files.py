#!/usr/bin/env python
#based on https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

import argparse
import numpy as np
import os
import tensorflow as tf

import six

from PIL import Image


from object_detection.utils import label_map_util #remember to compile Protobuf "protoc object_detection/protos/*.proto --python_out=."
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import dataset_util

label = [ 1, 2, 3 ,4]
label_description = [ 'RED', 'YELLOW', 'GREEN','' ]

def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = None # Image height
  width = None # Image width
  filename = None # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = None # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def encode_image_array_as_jpg_str(image):
  """Encodes a numpy array into a JPEG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    JPEG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='JPEG')
  jpg_string = output.getvalue()
  output.close()
  return jpg_string

def create_tf_example(box, classindex, image, filename):
    height = image.shape[0]
    width = image.shape[1]
    encoded_image_data = encode_image_array_as_jpg_str(image) # Encoded image bytes
    image_format = b'jpeg' # b'jpeg' or b'png'

    xmins = [ box[1] ] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [ box[3] ] # List of normalized right x coordinates in bounding box
                        # (1 per box)
    ymins = [ box[0] ] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [ box[2] ] # List of normalized bottom y coordinates in bounding box
                        # (1 per box)
    classes_text = [ label_description[classindex] ] # List of string class name of bounding box (1 per box)
    classes = [ label[classindex] ] # List of integer class id of bounding box (1 per box)

    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_record

def image_to_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating my own_dataset from rosbag')
    parser.add_argument('--path', type=str, default='', help='the path has images and labels.csv file')
    parser.add_argument('--output_file', type=str, default='test.record', help='output file name')
    args = parser.parse_args()
    csvfile = os.getcwd() + os.path.join(args.path,'labels.csv')

    # reading csvfile file
    imagefiles = []
    newlabels = []
    with open(csvfile) as f:
        i = 0
        for line in f:
            if i > 0:
                data = line.split(',')
                imagefiles.append(os.getcwd()+os.path.join(args.path,data[1]))
                newlabels.append(int(data[9]))
            i += 1

    # load the model into memory
    MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_PRETRAIN_LABELS = 'mscoco_label_map.pbtxt'
    #PATH_TO_PRETRAIN_LABELS = './models/object_detection/data/mscoco_label_map.pbtxt'
    NUM_CLASSES = 90
    TRAFFIC_LIGHT_CLASS = 10
    THRESHOLD = 0.89
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_PRETRAIN_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    writer = tf.python_io.TFRecordWriter(args.output_file)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for i in range(len(imagefiles)):
                if newlabels[i] < 3:     # avoid to clasify unknown
                    image = Image.open(imagefiles[i])
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = image_to_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    classes = np.squeeze(classes).astype(np.int32)
                    boxes = np.squeeze(boxes)
                    scores = np.squeeze(scores)
                    for j in range(len(classes)):
                        if classes[j] == TRAFFIC_LIGHT_CLASS:
                            if scores[j] is not None and scores[j] > THRESHOLD:
                                print "writing record:", i, imagefiles[i], boxes[j], "score:", scores[j], "new label:", newlabels[i]
                                record = create_tf_example(boxes[j], newlabels[i], image_np, imagefiles[i])
                                writer.write(record.SerializeToString())
    writer.close()


