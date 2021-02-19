# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 09:22:31 2020

@author: hp
"""

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import glob
import argparse
import threading
import six.moves.queue as Queue
import traceback
import numpy as np
import tensorflow as tf
import PIL.Image
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage


def rot90(v):
    return np.array([-v[1], v[0]])


def process_func(orig_path, landmarks):

    img = PIL.Image.open(orig_path)

    # Choose oriented crop rectangle.
    lm = landmarks
    eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5
    mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5
    eye_to_eye = lm[1] - lm[0]
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - rot90(eye_to_mouth)
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = rot90(x)
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    zoom = 1024 / (np.hypot(*x) * 2)

    # Shrink.
    shrink = int(np.floor(0.5 / zoom))
    if shrink > 1:
        size = (int(np.round(float(img.size[0]) / shrink)),
                int(np.round(float(img.size[1]) / shrink)))
        img = img.resize(size, PIL.Image.ANTIALIAS)
        quad /= shrink
        zoom *= shrink

    # Crop.
    border = max(int(np.round(1024 * 0.1 / zoom)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(
        np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Simulate super-resolution.
    superres = int(np.exp2(np.ceil(np.log2(zoom))))
    if superres > 1:
        img = img.resize(
            (img.size[0] * superres, img.size[1] * superres), PIL.Image.ANTIALIAS)
        quad *= superres
        zoom /= superres

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(
        np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] -
                                                                   img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.round(1024 * 0.3 / zoom)))
        img = np.pad(np.float32(img),
                     ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.mgrid[:h, :w, :1]
        mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad[0], np.float32(
            y) / pad[1]), np.minimum(np.float32(w-1-x) / pad[2], np.float32(h-1-y) / pad[3]))
        blur = 1024 * 0.02 / zoom
        img += (scipy.ndimage.gaussian_filter(img,
                                              [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(
            np.uint8(np.clip(np.round(img), 0, 255)), 'RGB')
        quad += pad[0:2]

    # Transform.
    img = img.transform((4096, 4096), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    img = img.resize((512, 512), PIL.Image.ANTIALIAS)
    img = np.asarray(img).transpose(0, 1, 2)
    return img
