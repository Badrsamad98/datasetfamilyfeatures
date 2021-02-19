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
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)
from lib.Image_creator import process_func

tf.compat.v1.enable_eager_execution()

flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')
flags.DEFINE_boolean('webcam', False, 'get image source from webcam or not')


rootdir = os.getcwd()+'\\downloads\\'
outputpath = os.getcwd()+'\\landmarks\\'

'''Creates the folder structure'''
def create_folder_structure(rootdir, outputpath):
    for subdir, dirs, files in os.walk(rootdir):
        structure = os.path.join(outputpath, subdir[len(rootdir):])
        if not os.path.isdir(structure):
            os.mkdir(structure)


'''Loads and trains the model'''
def initialize():
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    # Setup logger
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # Define network
    model = RetinaFaceModel(cfg, training=False,
                            iou_th=FLAGS.iou_th, score_th=FLAGS.score_th)

    # Load checkpoints
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)

    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print(
            "[*] load ckpt from {}.".format(tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    return model, cfg


'''Loops over all the image files to process them'''
def create_images(rootdir, outputpath, model, cfg):
    for subdir, dirs, files in os.walk(rootdir):
        f = open(os.path.join(subdir, 'landmarks.json'), "w")
        data = {"data": []}
        for file in files:
            try:
                if str(file).split('.')[-1] == 'jpg':
                    img_origin_path = os.path.join(subdir, file)
                    img_output_path = img_origin_path.replace(
                        rootdir, outputpath)
                    data = process_single_image(
                        img_origin_path, img_output_path, model, cfg, data)
            except:
                # manage excetions here
                print("Error ", sys.exc_info()[0], " occurred.")
        json.dump(data, f)


'''Processes a single image'''
def process_single_image(img_path, img_outputpath, model, cfg, data):

    if not os.path.exists(img_path):
        print(f"cannot find image path from {img_path}")
        exit()

    img_raw = cv2.imread(img_path)
    img_height_raw, img_width_raw, _ = img_raw.shape
    img = np.float32(img_raw.copy())

    if FLAGS.down_scale_factor < 1.0:
        img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                         fy=FLAGS.down_scale_factor,
                         interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pad input image to avoid unmatched shape problem
    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

    # Run model
    outputs = model(img[np.newaxis, ...]).numpy()
    # Recover padding effect
    outputs = recover_pad_output(outputs, pad_params)

    landmarks = []
    for prior_index in range(len(outputs)):
        x = draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw,
                            img_width_raw)
        landmarks.append(x)
    cv2.imwrite(img_outputpath, img_raw)

    return get_json_landmark_data(data, img_outputpath, landmarks)


'''Returns formatted json landmark data'''
def get_json_landmark_data(data, img_outputpath, landmarks):
    # Image name without the extension
    image_name = os.path.split(img_outputpath)[1].replace('.jpg', '')

    data["data"].append({
        image_name: []
    })

    # For all the detected faces, insert the corresponsing data in the json array
    for index, landmark in enumerate(landmarks):
        landmark = landmark.tolist()
        for e in data["data"]:
            if(image_name in e):
                e[image_name].append({
                    'lefteye': landmark[0],
                    'righteye': landmark[1],
                    'nose': landmark[2],
                    'leftmouthcorner': landmark[3],
                    'rightmouthcorner': landmark[4]
                })

    return data


'''Execute the script'''
def main(_argv):
    create_folder_structure(rootdir, outputpath)
    model, cfg = initialize()
    create_images(rootdir, outputpath, model, cfg)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
