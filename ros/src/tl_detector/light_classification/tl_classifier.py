from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy
import tensorflow as tf
import keras.backend as kb
from keras.preprocessing.image import load_img, img_to_array
from train import SqueezeNet
from consts import IMAGE_WIDTH, IMAGE_HEIGHT
from graph_utils import load_graph

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
		kb.set_image_dim_ordering('tf')

        self.ready = False

        if sim:
            model_name = 'squeezeNet_sim'
        else:
            model_name = 'squeezeNet_real'

        self.graph, ops = load_graph('light_classification/trained_model/{}.frozen.pb'.format(model_name))
        self.sess = tf.Session(graph = self.graph)

        self.learning_phase_tensor = self.graph.get_tensor_by_name('fire9_dropout/keras_learning_phase:0')
        self.op_tensor = self.graph.get_tensor_by_name('softmax/Softmax:0')
        self.input_tensor = self.graph.get_tensor_by_name('input_1:0')

        self.ready = True

        self.pred_dict = {0: TrafficLight.UNKNOWN,
                          1: TrafficLight.RED,
                          2: TrafficLight.GREEN}

        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN
