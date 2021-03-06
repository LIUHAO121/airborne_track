
from centernet.ctdet_detector import CtdetDetector

import random
import cv2
import numpy as np
from PIL import Image
from evaluator.airborne_detection import AirbornePredictor

import torch
from tqdm import tqdm

import os
from os import listdir
from os.path import isfile, join

import sys
sys.path.append("centernet")
sys.path.append("reid")
from centernet.ctdet_detector import CtdetDetector
from emb_tracker.get_feature import pth_extractor 
from emb_tracker.multi_tracker_single_match import EmbeddingTracker as SingleEBTracker
from emb_tracker.multi_tracker import EmbeddingTracker as EmbTracker

from utility.utils import read_yaml,py_cpu_nms,max_score_row,max_score_and_size




NMS_THRESH = 0.35
LINEAR_ASSIGNMENT_DIST = 0.3
FEAT_ALPHA = 0.05

MIN_SCORE = 0.5
TRACK_SCORE = 0.6
MIN_TRACK_LEN = 20
TRACK_LET_START = 0
FRAME_RATE=3
SPATIAL_DIST=2

class TrackByDetPredictor(AirbornePredictor):
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
        model_path = os.path.join(current_path, 'centernet/model_res18_png123pre16_e36_2048.pth')
        opts = read_yaml(config_file)
        opts['load_model'] = model_path
        self.tracker = SingleEBTracker(conf_thres=TRACK_SCORE,feat_alpha=FEAT_ALPHA,frame_rate=FRAME_RATE,linear_assignment_dist=LINEAR_ASSIGNMENT_DIST,spatial_dist=SPATIAL_DIST)
        self.ctdetect = CtdetDetector(opts)
        self.reid_feat = pth_extractor()
        

    def make_track_input(self,img,dets):
        if len(dets)>0:
            nms_result = py_cpu_nms(np.array(dets),NMS_THRESH)
            dets = [dets[i] for i in nms_result]
        all_det_data = []
        for det in dets:
            det_data = {}
            x1,y1,x2,y2 = list(map(int,det[:4]))

            x1 = max(0,x1)
            x2 = max(0,x2)
            y1 = max(0,y1)
            y2 = max(0,y2)

            w = x2 - x1
            h = y2 - y1
            if w < 3 or h < 3:
                continue
            score = det[-1]    
            det_data['emb'] = self.reid_feat(img[y1:y2,x1:x2,:]).squeeze().tolist()
 
            det_data["conf"] = score
            det_data["bbox_tlwh"] = [x1,y1,x2-x1,y2-y1]
            all_det_data.append(det_data)
        return all_det_data


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
            for entity in self.track_id_results[track_id][TRACK_LET_START:]:
                if entity[3] < MIN_SCORE:
                    continue
                # self.register_object_and_location(*entity)
                self.register_object_and_location(class_name=entity[0],track_id=1,bbox=entity[2],confidence=entity[3],img_name=entity[4])

    """
    PARTICIPANT_TODO:
    During the evaluation all combinations for flight_id and flight_folder_path will be provided one by one.
    """
    def inference(self, flight_id):
        self.flight_started()
        self.tracker.init_history_state()
        for frame_image in tqdm(self.get_all_frame_images(flight_id)):
            frame_image_path = self.get_frame_image_location(flight_id, frame_image)
            frame = cv2.imread(frame_image_path)
            
            dets = self.ctdetect.run(frame)

            class_name = 'airborne'

            select_dets = []
            for category in list(dets.keys()):
                if int(category) in [1,2,3,4]:
                    c_dets = dets[category]
                    for det in c_dets:
                        if det[-1] > MIN_SCORE:
                            select_dets.append(det)

            # select_dets = max_score_row(select_dets)   
            select_dets = max_score_and_size(select_dets)
            
            track_input = self.make_track_input(frame,select_dets)
            track_output = self.tracker.update(track_input)
            for track in track_output:
                bbox = list(map(float,track.tlbr))
                confidence = track.score
                track_id  = track.track_id
                self.proxy_register_object_and_location(class_name, int(track_id), 
                                                bbox, float(confidence), 
                                                frame_image)

        self.flight_completed()

if __name__ == "__main__":
    submission = TrackByDetPredictor()
    submission.run()