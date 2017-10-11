### Team Members

|Name                |Email Address                  |Main Role             |
|--------------------|-------------------------------|----------------------|
|Samer Abbas         |samer1abbas@gmail.com          |Team Lead/ DBW Node   |
|John Carpenter      |jcarpent@gmail.com             |TL Detection          |
|Erick Ramirez       |erickramireztebalan@gmail.com  |TL Detection          |
|Innocent Djiofack   |djiofack007@gmail.com          |Waypoint Updater      |
|Mohamed Saleh       |maldwaib@gmail.com             |Waypoint Updater      |

### Notes to Reviewers:

* Please download the traffic light detection model from: https://s3-us-west-2.amazonaws.com/sdcnd/tl_detector.zip
* If you want to see the camera data show on screen for the real-world data you have to update the file `ros/src/tl_detector/tl_detector.py` by changing the line: `enable_imshow = False` to True.
* We used the NN solution for real world data and the CV solution for the simulator.  The launch file `/ros/src/tl_detector/launch`
has a parameter for this:   `<param name="UseNN" value="False" />` where the parameter can be changed to switch between CV and NN. The file `ros/src/tl_detector/tl_detector.py` has the following code that picks between VC and NN depending on the above mentioned parameter:
```
UseNN = rospy.get_param('UseNN')
	if UseNN ==True:
		self.light_classifier = TLClassifier(os.path.dirname(__file__) + '/light_classification/faster-R-CNN')
	else:
	        self.light_classifier = TLClassifierCV()
        self.listener = tf.TransformListener()
```

## Simulator with NN
[![](https://img.youtube.com/vi/WzAnCp_eU-o/0.jpg)](https://youtu.be/WzAnCp_eU-o "Simulator and NN")

## Simulator with CV
[![](https://img.youtube.com/vi/I1V4cbvkzpM/0.jpg)](https://youtu.be/I1V4cbvkzpM "Simulator and CV")

## Real World and NN
[![](https://img.youtube.com/vi/QbC0nYbFq3U/0.jpg)](https://youtu.be/QbC0nYbFq3U "Real World and NN")

## Real World and CV
[![](https://img.youtube.com/vi/xQ8Tdn8JiXs/0.jpg)](https://youtu.be/xQ8Tdn8JiXs "Real World and CV")
