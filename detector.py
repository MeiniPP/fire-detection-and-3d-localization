"""Original YOLOv5 ONNX decoding and HSV filtering shared by both entry points."""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

DEFAULT_MODEL = Path(__file__).absolute().parent / 'models' / 'best.onnx'


class FireDetector:
    """Load the project's raw, single-class YOLOv5 ONNX export."""

    def __init__(self, model=DEFAULT_MODEL, confidence=0.5, nms=0.4, provider='cpu'):
        available = ort.get_available_providers()
        if provider == 'cuda' and 'CUDAExecutionProvider' not in available:
            raise RuntimeError('CUDA is unavailable. Install a compatible onnxruntime-gpu build or use --provider cpu.')
        providers = ['CPUExecutionProvider']
        if provider == 'cuda' or (provider == 'auto' and 'CUDAExecutionProvider' in available):
            providers.insert(0, 'CUDAExecutionProvider')
        self.net = ort.InferenceSession(str(model), providers=providers)
        if provider == 'cuda' and 'CUDAExecutionProvider' not in self.net.get_providers():
            raise RuntimeError('CUDA initialization failed; check the CUDA and cuDNN dependencies.')
        shape = self.net.get_inputs()[0].shape
        if shape != [1, 3, 320, 320] or self.net.get_outputs()[0].shape != [1, 6300, 6]:
            raise ValueError('Expected the bundled raw YOLOv5 export: input [1,3,320,320], output [1,6300,6].')
        self.confidence = confidence
        self.nms = nms
        self.anchors = np.asarray([[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
                                   [116, 90, 156, 198, 373, 326]], dtype=np.float32).reshape(3, -1, 2)

    def detect(self, frame, hsv_filter=False):
        boxes, scores, classes = infer_img(frame, self.net, 320, 320, 3, 3,
                                           [8., 16., 32.], self.anchors, self.nms, self.confidence)
        height, width = frame.shape[:2]
        detections = []
        for box, score, class_id in zip(boxes, scores, classes):
            left, top, right, bottom = np.asarray(box, dtype=np.int32)
            left, right = max(0, min(width, left)), max(0, min(width, right))
            top, bottom = max(0, min(height, top)), max(0, min(height, bottom))
            if right <= left or bottom <= top:
                continue
            if hsv_filter and not is_fire_hsv(cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)):
                continue
            detections.append(((left, top, right, bottom), float(score), int(class_id)))
        return detections

def _make_grid(nx, ny):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

def cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride):
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = (int(model_w / stride[i]), int(model_h / stride[i]))
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)
        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2.0 - 0.5 + np.tile(grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs

def post_process_opencv(outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
    conf = outputs[:, 4].tolist()
    c_x = outputs[:, 0] / model_w * img_w
    c_y = outputs[:, 1] / model_h * img_h
    w = outputs[:, 2] / model_w * img_w
    h = outputs[:, 3] / model_h * img_h
    p_cls = outputs[:, 5:]
    if len(p_cls.shape) == 1:
        p_cls = np.expand_dims(p_cls, 1)
    cls_id = np.argmax(p_cls, axis=1)
    p_x1 = np.expand_dims(c_x - w / 2, -1)
    p_y1 = np.expand_dims(c_y - h / 2, -1)
    p_x2 = np.expand_dims(c_x + w / 2, -1)
    p_y2 = np.expand_dims(c_y + h / 2, -1)
    areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)
    areas = areas.tolist()
    nms_boxes = np.column_stack((p_x1[:, 0], p_y1[:, 0], w, h)).tolist()
    ids = np.asarray(cv2.dnn.NMSBoxes(nms_boxes, conf, thred_cond, thred_nms)).reshape(-1)
    if len(ids) > 0:
        return (np.array(areas)[ids], np.array(conf)[ids], cls_id[ids])
    else:
        return ([], [], [])

def infer_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4, thred_cond=0.5):
    img = cv2.resize(img0, (model_w, model_h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
    outs = cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride)
    img_h, img_w, _ = np.shape(img0)
    boxes, confs, ids = post_process_opencv(outs, model_h, model_w, img_h, img_w, thred_nms, thred_cond)
    return (boxes, confs, ids)

def is_fire_hsv(hsv_image):
    lower_fire_yellow_orange = np.array([10, 100, 100])
    upper_fire_yellow_orange = np.array([35, 255, 255])
    lower_fire_red1 = np.array([10, 100, 100])
    upper_fire_red1 = np.array([25, 255, 255])
    lower_fire_red2 = np.array([160, 100, 100])
    upper_fire_red2 = np.array([180, 255, 255])
    mask_yellow_orange = cv2.inRange(hsv_image, lower_fire_yellow_orange, upper_fire_yellow_orange)
    mask_red1 = cv2.inRange(hsv_image, lower_fire_red1, upper_fire_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_fire_red2, upper_fire_red2)
    mask = cv2.bitwise_or(mask_yellow_orange, cv2.bitwise_or(mask_red1, mask_red2))
    if cv2.countNonZero(mask) > 0.001 * mask.size:
        return True
    return False
