
from centernet.ctdet_detector import CtdetDetector

import random
import cv2
from PIL import Image
from evaluator.airborne_detection import AirbornePredictor

import torch
from tqdm import tqdm

import os
from os import listdir
from os.path import isfile, join

import sys
sys.path.append("centernet")
from centernet.ctdet_detector import CtdetDetector
from utility.utils import read_yaml


MIN_TRACK_LEN = 30
MIN_SCORE = 0.5


class CtPredictor(AirbornePredictor):
    """
    PARTICIPANT_TODO: You can name your implementation as you like. `RandomPredictor` is just an example.
    Below paths will be preloaded for you, you can read them as you like.
    """
    training_data_path = None
    test_data_path = None
    vocabulary_path = None

    """
    PARTICIPANT_TODO:
    You can do any preprocessing required for your codebase here like loading up models into memory, etc.
    """
    def inference_setup(self):
        current_path = os.getcwd()
        config_file = os.path.join(current_path, 'centernet/airborne_config.yaml')
        model_path = os.path.join(current_path, 'centernet/model_res18_e13_2048.pth')
        opts = read_yaml(config_file)
        opts['load_model'] = model_path
        self.ctdetect = CtdetDetector(opts)

    def get_all_frame_images(self, flight_id):
        frames = []
        flight_folder = join(self.test_data_path, flight_id)
        for frame in sorted(listdir(flight_folder)):
            if isfile(join(flight_folder, frame)):
                frames.append(frame)
        return frames

    def flight_started(self):
        self.track_id_results = {}
        self.visited_frame = {}

    def proxy_register_object_and_location(self, class_name, track_id, bbox, confidence, img_name):
        if track_id not in self.track_id_results:
            self.track_id_results[track_id] = []
        if img_name not in self.visited_frame:
            self.visited_frame[img_name] = []

        if track_id in self.visited_frame[img_name]:
            raise Exception('two entities  within the same frame {} have the same track id'.format(img_name))

        self.track_id_results[track_id].append([class_name, track_id, bbox, confidence, img_name])
        self.visited_frame[img_name].append(track_id)

    def flight_completed(self):
        for track_id in self.track_id_results.keys():
            track_len = len(self.track_id_results[track_id])
            if track_len < MIN_TRACK_LEN:
                continue
            for entity in self.track_id_results[track_id][MIN_TRACK_LEN:]:
                if entity[3] < MIN_SCORE:
                    continue
                self.register_object_and_location(*entity)

    """
    PARTICIPANT_TODO:
    During the evaluation all combinations for flight_id and flight_folder_path will be provided one by one.
    """
    def inference(self, flight_id):
        self.flight_started()

        for frame_image in tqdm(self.get_all_frame_images(flight_id)):
            frame_image_path = self.get_frame_image_location(flight_id, frame_image)
            frame = cv2.imread(frame_image_path)
            
            results = self.ctdetect.run(frame)

            class_name = 'airborne'

            select_dets = []
            for category in list(results.keys()):
                if int(category) in [1,2,3,4]:
                    c_dets = results[category]
                    for det in c_dets:
                        if det[-1] > MIN_SCORE:
                            select_dets.append(det)

            for track_id,det in enumerate(select_dets):
                bbox = list(map(float,det[:4]))
                confidence = det[-1]
                self.proxy_register_object_and_location(class_name, int(track_id), 
                                                bbox, float(confidence), 
                                                frame_image)

        self.flight_completed()

if __name__ == "__main__":
    submission = CtPredictor()
    submission.run()