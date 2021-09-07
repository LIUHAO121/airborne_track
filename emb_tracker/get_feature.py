import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import time
from .deepsort_model import Net

class pth_extractor(object):
    def __init__(self, model_path='emb_tracker/ckpt_reid.pth', use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)[
            'net_dict']
        self.net.load_state_dict(state_dict)
        print("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.net.eval()
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, img):
        if isinstance(img,str):
            im = cv2.imread(img)
        elif isinstance(img,np.ndarray):
            im = img
        else:
            print("unrecognize input:",img)
        im = im[:, :, (2, 1, 0)]
        im_crops = [im]
        def _resize(im, size):
            return (cv2.resize(im, size)).astype(np.float32)/255.
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im):
        im_batch = self._preprocess(im)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        feat = features.cpu().numpy()
        feat = feat.squeeze()
        feat /= np.linalg.norm(feat)
        feat = feat.reshape(1, -1)
        # feat = torch.from_numpy(feat)
        return feat
