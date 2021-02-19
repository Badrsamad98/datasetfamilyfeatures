from PIL import Image
from pathlib import Path
import shutil
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import sys
import os
import numpy as np
import tensorflow as tf
import time
import json
from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml,
                           draw_bbox_landm, pad_input_image, recover_pad_output)
from lib.Image_creator import process_func

rootdir = os.getcwd()+'\\downloads\\'
outputpath = os.getcwd()+'\\output\\'

'''Creates the folder structure'''
def create_folder_structure(rootdir, outputpath):
    for subdir, dirs, files in os.walk(rootdir):
        structure = os.path.join(outputpath, subdir[len(rootdir):])
        if not os.path.isdir(structure):
            os.mkdir(structure)


'''Processes all the images to crop'''
def parse_images_to_crop(rootdir, outputpath):
    for subdir, dirs, files in os.walk(rootdir):
        landmarks = load_landmarks_in_subdirectory(subdir)
        try:
            for file in os.listdir(subdir):
                if os.path.isfile(os.path.join(subdir, file)):
                    if(file != "landmarks.json" and file.split('.')[1] != 'csv'):
                        # Process all images
                        process_crop_copy_all_faces(
                            file, landmarks, subdir, rootdir, outputpath)
                    else:
                        # Copy all other files
                        path = os.path.join(subdir, file)
                        shutil.copy(path, path.replace(rootdir, outputpath))
        except:
            print("Error ", sys.exc_info()[0], " occurred.")


'''Loads the json landmark files'''
def load_landmarks_in_subdirectory(subdir):
    print(subdir)
    if(os.path.isfile(os.path.join(subdir, 'landmarks.json'))):
        with open(os.path.join(subdir, 'landmarks.json'), 'r') as f:
            return json.loads(f.read())


'''Processes, crops and copies one images resulting into perheaps multiple faces'''
def process_crop_copy_all_faces(file, landmarks, subdir, rootdir, outputpath):
    # Name of the image without the extension to load the landmarks
    key = file.replace('.jpg', '')
    # Get the landmarks in the json file
    landmark = [obj for obj in landmarks["data"] if key in obj]

    for index, face in enumerate(landmark[0][key]):
        # For each face, get the position of the features
        face_array = np.array([face["lefteye"], face["righteye"], face
                               ["nose"], face["leftmouthcorner"], face["rightmouthcorner"]])

        img_path = os.path.join(subdir, file)
        # Use the process function to prepare the image and crop it with the list of landmarks
        image = process_func(img_path, face_array)

        # If one image, copy without rename, else rename with index
        if(len(landmark[0][key]) == 1):
            path = os.path.join(subdir.replace(rootdir, outputpath), file)
        else:
            path = os.path.join(subdir.replace(
                rootdir, outputpath), file.replace('.jpg', '_'+str(index)+'.jpg'))

        cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, cvt)


'''Execute the script'''
create_folder_structure(rootdir, outputpath)
parse_images_to_crop(rootdir, outputpath)
