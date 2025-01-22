from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, \
    encode_box_info
# from .utils_vctree import generate_forest, arbForest_to_biForest, get_overlap_info, rebuild_forest
# from .utils_treelstm import TreeLSTM_IO, MultiLayer_BTreeLSTM, BiTreeLSTM_Backward, BiTreeLSTM_Foreward
from .utils_relation import layer_init
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import scipy.stats
import random


def encode_box_info(proposals):
    """
    encode proposed box information (x1, y1, x2, y2) to
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    assert proposals[0].mode == 'xyxy'
    boxes_info = []
    for proposal in proposals:
        boxes = proposal.bbox
        img_size = proposal.size
        wid = img_size[0]
        hei = img_size[1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        w, h = wh.split([1, 1], dim=-1)
        x, y = xy.split([1, 1], dim=-1)
        x1, y1, x2, y2 = boxes.split([1, 1, 1, 1], dim=-1)
        assert wid * hei != 0
        info = torch.cat([w / wid, h / hei, x / wid, y / hei, x1 / wid, y1 / hei, x2 / wid, y2 / hei,
                          w * h / (wid * hei)], dim=-1).view(-1, 9)
        boxes_info.append(info)

    return torch.cat(boxes_info, dim=0)

# 计算候选框和真实框之间的位置偏移量，以及尺度变化的目标
def bbox_transform_inv(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths # 中心坐标
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.stack((targets_dx, targets_dy, targets_dw,
                           targets_dh), -1)
    return targets


def get_spt_features(boxes1, boxes2, boxes_u, width, height):
    # boxes_u = boxes_union(boxes1, boxes2)
    spt_feat_1 = get_box_feature(boxes1, width, height)
    spt_feat_2 = get_box_feature(boxes2, width, height)
    spt_feat_12 = get_pair_feature(boxes1, boxes2)
    spt_feat_1u = get_pair_feature(boxes1, boxes_u)
    spt_feat_u2 = get_pair_feature(boxes_u, boxes2)
    return torch.cat((spt_feat_12, spt_feat_1u, spt_feat_u2, spt_feat_1, spt_feat_2), -1)

# 只嵌入主体空间特征和联合特征
def get_spt_su_features(boxes1, boxes_u, width, height):
    # boxes_u = boxes_union(boxes1, boxes2)
    spt_feat_1 = get_box_feature(boxes1, width, height)
    spt_feat_1u = get_pair_feature(boxes1, boxes_u)
    return torch.cat((spt_feat_1u, spt_feat_1), -1)

# 只嵌入客体空间特征和联合特征
def get_spt_uo_features(boxes2, boxes_u, width, height):
    # boxes_u = boxes_union(boxes1, boxes2)
    spt_feat_2 = get_box_feature(boxes2, width, height)
    spt_feat_2u = get_pair_feature(boxes2, boxes_u)
    return torch.cat((spt_feat_2u, spt_feat_2), -1)


def get_area(boxes):
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return area

# 这个函数的主要目的是计算一对候选框之间的特征表示，可以用于目标检测中的一些任务，如关系检测、配对任务等。通过计算候选框之间的位置偏移和尺度变化，可以提取有关候选框对的空间关系特征，进而用于后续的任务处理。
def get_pair_feature(boxes1, boxes2):
    delta_1 = bbox_transform_inv(boxes1, boxes2)# 将 boxes1 转换为 boxes2 的回归目标
    delta_2 = bbox_transform_inv(boxes2, boxes1)# 将 boxes2 转换为 boxes1 的回归目标
    spt_feat = torch.cat((delta_1, delta_2[:, :2]), -1)
    return spt_feat


def get_box_feature(boxes, width, height): # 候选框的坐标归一化
    f1 = boxes[:, 0] / width
    f2 = boxes[:, 1] / height
    f3 = boxes[:, 2] / width
    f4 = boxes[:, 3] / height
    f5 = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1) / (width * height)
    return torch.stack((f1, f2, f3, f4, f5), -1)


class Boxes_Encode(nn.Module):
    def __init__(self, ):
        super(Boxes_Encode, self).__init__()
        self.spt_feats = nn.Sequential(
            nn.Linear(28, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1))

    def spo_boxes(self, boxes, rel_inds):
        # boxes 表示对象的位置信息
        # rel_inds 表示关系的信息
        # 从位置信息中获取主语的位置信息
        s_boxes = boxes[rel_inds[:, 0]]
        # 从位置信息中获取宾语的位置信息
        o_boxes = boxes[rel_inds[:, 1]]
        # 计算主语和宾语位置的并集，即获取两个对象之间的联合框
        union_boxes = torch.cat((
            torch.min(s_boxes[:, 0:2], o_boxes[:, 0:2]),
            torch.max(s_boxes[:, 2:], o_boxes[:, 2:])
        ), 1)

        return s_boxes, o_boxes, union_boxes

    def forward(self, boxes, rel_inds, width, height):
        s_boxes, o_boxes, u_boxes = self.spo_boxes(boxes, rel_inds)
        spt_feats = get_spt_features(s_boxes, o_boxes, u_boxes, width, height)

        return self.spt_feats(spt_feats)

class Boxes_su_Encode(nn.Module):
    def __init__(self, ):
        super(Boxes_su_Encode, self).__init__()
        self.spt_su_feats = nn.Sequential(
            nn.Linear(11, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1)
        )

    def spo_boxes(self, boxes, rel_inds):
        # boxes 表示对象的位置信息
        # rel_inds 表示关系的信息
        # 从位置信息中获取主语的位置信息
        s_boxes = boxes[rel_inds[:, 0]]
        # 从位置信息中获取宾语的位置信息
        o_boxes = boxes[rel_inds[:, 1]]
        # 计算主语和宾语位置的并集，即获取两个对象之间的联合框
        union_boxes = torch.cat((
            torch.min(s_boxes[:, 0:2], o_boxes[:, 0:2]),
            torch.max(s_boxes[:, 2:], o_boxes[:, 2:])
        ), 1)
        # print("union_boxes的值", union_boxes)
        # print("union_boxes的形状", union_boxes.shape)  # shape属性显示张量形状
        # print('*' * 100)

        return s_boxes, o_boxes, union_boxes

    def forward(self, boxes, rel_inds, width, height):
        s_boxes, o_boxes, u_boxes = self.spo_boxes(boxes, rel_inds)
        spt_su_feats = get_spt_su_features(s_boxes, u_boxes, width, height)
        # print("spt_su_feats的值", spt_su_feats)
        # print("spt_su_feats的形状", spt_su_feats.shape)  # shape属性显示张量形状
        # print('*' * 100)
        return self.spt_su_feats(spt_su_feats)

class Boxes_uo_Encode(nn.Module):
    def __init__(self, ):
        super(Boxes_uo_Encode, self).__init__()
        self.spt_uo_feats = nn.Sequential(
            nn.Linear(11, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1)
        )

    def spo_boxes(self, boxes, rel_inds):
        # boxes 表示对象的位置信息
        # rel_inds 表示关系的信息
        # 从位置信息中获取主语的位置信息
        s_boxes = boxes[rel_inds[:, 0]]
        # 从位置信息中获取宾语的位置信息
        o_boxes = boxes[rel_inds[:, 1]]
        # 计算主语和宾语位置的并集，即获取两个对象之间的联合框
        union_boxes = torch.cat((
            torch.min(s_boxes[:, 0:2], o_boxes[:, 0:2]),
            torch.max(s_boxes[:, 2:], o_boxes[:, 2:])
        ), 1)
        # print("union_boxes的值", union_boxes)
        # print("union_boxes的形状", union_boxes.shape)  # shape属性显示张量形状
        # print('*' * 100)

        return s_boxes, o_boxes, union_boxes

    def forward(self, boxes, rel_inds, width, height):
        s_boxes, o_boxes, u_boxes = self.spo_boxes(boxes, rel_inds)
        spt_uo_feats = get_spt_su_features(o_boxes, u_boxes, width, height)
        return self.spt_uo_feats(spt_uo_feats)


