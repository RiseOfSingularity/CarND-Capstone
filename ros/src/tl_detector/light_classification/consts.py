#Based on:
#https://github.com/mynameisguy/TrafficLightChallenge-DeepLearning-Nexar (Guy Hadash)
#Using  SqueezeNet architecture - https://github.com/DeepScale/SqueezeNet

## dirs confs
DATASET_FOLDER="./traffic_light_bag_files"
MODELS_CHECKPOINTS_DIR="./checkpoints"

# training confs
BATCH_SIZE = 64
TRAINING_EPOCHS = 200 #max
TRAIN_IMAGES_PER_EPOCH = 16768
VALIDATE_IMAGES_PER_EPOCH = 1856
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224