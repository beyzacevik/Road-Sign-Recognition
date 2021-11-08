import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from RoadSignRecognition.ColorDetection import *
from RoadSignRecognition.ArrowDirectionDetection import *
from RoadSignRecognition.ShapeDetection import *

def detect_road_signs(directory_path):

    for num, filename in enumerate(os.listdir(directory_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):

            img_path = os.path.join(directory_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_temp = find_most_common_color(img_path)
            color = get_color_name(img_temp)
            shapes, img_shapes = detect_shapes(img_path)

            direction = arrow_detection_with_hough_lines(img_path)
            # general information about traffic sign
            info = 'This %s road sign formed of %s. ' % (color, ','.join(list(set(shapes))))

            # give the exact description if possible

            if direction:

                if color == 'blue':

                    if direction in ['left', 'right']:
                        info += '\n This is a turn %s sign.' % (direction)

                    else:
                        info += '\n This is a go %s sign.' % (direction)


                else:
                    info += '\n This road sign has an arrow-like shape.'
                    info += '\nThis road sign might contain numerical values such as a speed limit.'

            else:
                if color == 'red':
                    if 'rectangle' in shapes:
                        info += '\n This is a stop sign.'

            plt.title(info)
            plt.axis('off')
            plt.imshow(img)
            # plt.savefig('outputPath'+filename)



        else:
            continue


if __name__ == 'main':
    directory_path = ''
    detect_road_signs(directory_path)