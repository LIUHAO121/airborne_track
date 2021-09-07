import torch
import numpy as np
import cv2
from .utils import create_model,load_model,get_affine_transform,ctdet_decode,ctdet_post_process

class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)


class BaseDetector(object):
    def __init__(self, opt):
        opt.device = torch.device('cuda')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True


    def pre_process(self, image, meta=None):
        height, width = image.shape[0:2]
        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s, 
                'out_height': inp_height // self.opt.down_ratio, 
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta


    def process(self, images, return_time=False):
        raise NotImplementedError


    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError


    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):

        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type (''): 
            image = cv2.imread(image_or_path_or_tensor)

        images, meta = self.pre_process(image,meta)
        images = images.to(self.opt.device)
        dets  = self.process(images, return_time=True)
        dets = self.post_process(dets, meta) 
        return dets


class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        if isinstance(opt, dict):
            opt = Struct(opt)
        super(CtdetDetector, self).__init__(opt)
  
    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            dets = ctdet_decode(hm, wh, reg=reg) # (b,k,6) x1,y1,x2,y2,s,c
        return dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes) # [{1:[],2:[],3:[],,,}]
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0] # {1:[],2:[],3:[],,,}

    def draw_det(self,img,det_result,thresh=0.3,out_name="1.jpg"):
        im = np.ascontiguousarray(np.copy(img))
        im_h, im_w = im.shape[:2]
        line_thickness = max(1, int(img.shape[1] / 500.0))
        text_scale = 2
        text_thickness = 2
        for category in list(det_result.keys()):
            c_dets = det_result[category]
            for det in c_dets:
                if det[-1]>thresh:
                    det = list(map(int,det))
                    cv2.rectangle(im, tuple(det[0:2]), tuple(det[2:4]), color=(255,0,0), thickness=line_thickness)
                    cv2.putText(im, str(category), (det[0],det[1]-1), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
                    cv2.putText(im, str(det[-1])[:5], (det[0]+1,det[1]-1), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        cv2.imwrite(out_name,im)

