import time

import numpy as np
from typing import Dict, Any

from .utils.infer_engine import OrtInferSession
from .utils.load_image import LoadImage
from .utils.transform import (
    custom_NMSBoxes,
    ResizePad,
)

class Det:
    def __init__(self, config: Dict[str, Any]):
        self.model = OrtInferSession(config)
        self.img_loader = LoadImage()
        self.resize_shape = [928, 928]

    def __call__(self, img, **kwargs):
        start = time.time()
        score = kwargs.get("score", 0.4)
        img = self.img_loader(img)
        ori_h, ori_w = img.shape[:-1]
        img, new_w, new_h, left, top = self.img_preprocess(img, self.resize_shape)
        pre = self.model([img])
        result = self.img_postprocess(
            pre, ori_w / new_w, ori_h / new_h, left, top, score
        )
        return result, time.time() - start

    def img_preprocess(self, img, resize_shape=[928, 928]):
        im, new_w, new_h, left, top = ResizePad(img, resize_shape[0])
        im = im / 255.0
        im = im.transpose((2, 0, 1)).copy()
        im = im[None, :].astype("float32")
        return im, new_w, new_h, left, top

    def img_postprocess(self, predict_maps, x_factor, y_factor, left, top, score):
        result = []
        # 转置和压缩输出以匹配预期的形状
        outputs = np.transpose(np.squeeze(predict_maps[0]))
        # 获取输出数组的行数
        rows = outputs.shape[0]
        # 用于存储检测的边界框、得分和类别ID的列表
        boxes = []
        scores = []
        # 遍历输出数组的每一行
        for i in range(rows):
            # 找到类别得分中的最大得分
            max_score = outputs[i][4]
            # 如果最大得分高于置信度阈值
            if max_score >= score:
                # 从当前行提取边界框坐标
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # 计算边界框的缩放坐标
                xmin = max(int((x - w / 2 - left) * x_factor), 0)
                ymin = max(int((y - h / 2 - top) * y_factor), 0)
                xmax = xmin + int(w * x_factor)
                ymax = ymin + int(h * y_factor)
                # 将类别ID、得分和框坐标添加到各自的列表中
                boxes.append([xmin, ymin, xmax, ymax])
                scores.append(max_score)
                # 应用非最大抑制过滤重叠的边界框
        indices = custom_NMSBoxes(boxes, scores)
        for i in indices:
            result.append([scores[i], np.array(boxes[i])])
        return result

