from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset

import glob
import os
import os.path as osp

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
import numpy as np

@ROTATED_DATASETS.register_module()
class CylinderDataset(DOTADataset):
    """SAR ship dataset for detection."""
    CLASSES = ['cylinder']
    PALETTE = [(165, 42, 42)]
    
    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.jpg')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        x1, y1, w1, h1, a1, t1, x2, y2, w2, h2, a2, t2 = map(float, bbox_info)                      
                        gt_bboxes.append([x1, y1, w1, h1, a1, t1, x2, y2, w2, h2, a2, t2])
                        # gt_bboxes.append([x1, y1, w1, h1, a1, t1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

                        cls_name = 'cylinder'
                        label = cls_map[cls_name]
                        gt_labels.append(label)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos