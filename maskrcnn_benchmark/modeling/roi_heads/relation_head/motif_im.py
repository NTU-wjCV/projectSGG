import numpy as np
import torch
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
from torch.nn import functional as F
from scipy.spatial.distance import cdist

from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.layers import Label_Smoothing_Regression
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.modeling.utils import cat
from .model_gpsnet import Boxes_Encode, Boxes_su_Encode, Boxes_uo_Encode
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_msg_passing import IMPContext
from .model_transformer import TransformerContext
from .model_vctree import VCTreeLSTMContext
from .model_vtranse import VTransEFeature
from .utils_motifs import rel_vectors, obj_edge_vectors, nms_overlaps
from .utils_motifs import to_onehot, encode_box_info
from .utils_relation import layer_init, get_box_info, get_box_pair_info

class SequencePooling(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool = nn.AvgPool1d(pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x_pooled = self.pool(x)
        x_pooled = x_pooled[:, :, :-1] if x_pooled.shape[2] > x.shape[2] else x_pooled  # 保证池化后的长度与输入长度一致
        x = x_pooled - x
        return x.transpose(1, 2)

class SequenceMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 8  # 减少 MLP 隐藏层的维度
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SequencePoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=2, mlp_ratio=1., act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0.1,
                 drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = SequencePooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SequenceMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, importance_scores):
        B, N, C = x.shape  # 获取输入的形状
        x = x.reshape(-1, C)  # 将输入重塑为 (B*N, C)
        x = x.reshape(B, N, -1)  # 将输入恢复为 (B, N, C)

        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.reshape(-1, x.shape[-1])  # 将输入重塑为 (B*N, C)
        x = x.reshape(B, N, -1)  # 将输入恢复为 (B, N, C)

        # 使用重要性分数进行加权
        if importance_scores is not None:
            importance_scores = importance_scores.unsqueeze(0).unsqueeze(2).expand(B, N, C)
            x = x * (1 + importance_scores)

        return x

class SequencePoolFormer(nn.Module):
    def __init__(self, embed_dim=2048, depth=2):  # 将深度减少为3
        super(SequencePoolFormer, self).__init__()
        self.blocks = nn.ModuleList([
            SequencePoolFormerBlock(embed_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, importance_scores):
        for blk in self.blocks:
            x = blk(x, importance_scores)
        x = self.norm(x)  # 使用 4096 维度的归一化
        return x

@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor_im")
class MotifPredictor_im(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor_im, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        self.mlp_dim = 2048
        dropout_p = 0.2
        self.norm_pair1 = nn.Sequential(*[
            nn.Linear(2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.mlp_dim // 2),
            nn.Dropout(dropout_p),
            nn.BatchNorm1d(self.mlp_dim // 2)
        ])
        self.norm_pair2 = nn.Sequential(*[
            nn.Linear(2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.mlp_dim // 2),
            nn.Dropout(dropout_p),
            nn.BatchNorm1d(self.mlp_dim // 2)
        ])
        self.poolformer_attention = SequencePoolFormer(embed_dim=self.mlp_dim // 2, depth=2)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, rel_importance,
                rel_obj_classes
                , logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}
        initial_pair_pred_rep = torch.zeros(pair_pred.size(0), self.mlp_dim, device=pair_pred.device)
        initial_rel_obj_classes_rep = torch.zeros(len(rel_obj_classes), self.mlp_dim,
                                                  device=pair_pred.device) if rel_obj_classes is not None else None
        pair_pred_rep = initial_pair_pred_rep
        rel_obj_classes_rep = initial_rel_obj_classes_rep

        if rel_labels is None:
            rel_labels = [torch.tensor([]).cuda()]  # 假设你在使用GPU，如果使用CPU，去掉 .cuda()
        else:
            rel_labels = [torch.tensor(label).cuda() for label in rel_labels]  # 假设 rel_labels 已经是一个列表，并且你在使用GPU

        if rel_importance is None:
            rel_importance = torch.zeros(len(rel_labels)).cuda()
        else:
            rel_importance = [torch.tensor(ri) for ri in rel_importance]
            rel_importance = torch.cat(rel_importance).cuda()  # 假设你在使用GPU，如果使用CPU，去掉 .cuda()

        def compute_euclidean_distance(x1, x2):
            diff = x1 - x2
            dist_sq = torch.sum(diff ** 2, dim=-1)
            dist = torch.sqrt(dist_sq)
            return dist

        if self.training:
            if rel_obj_classes is not None:
                num_pairs = pair_pred.size(0)
                pair_pred_rep = self.norm_pair1(pair_pred.float())
                rel_obj_classes = torch.cat(rel_obj_classes, dim=0)

                # 确保 rel_obj_classes 的 size(0) 与 num_pairs 一致
                if rel_obj_classes.size(0) < num_pairs:
                    repeat_factor = (num_pairs // rel_obj_classes.size(0)) + 1
                    rel_obj_classes = rel_obj_classes.repeat(repeat_factor, 1)[:num_pairs]
                else:
                    rel_obj_classes = rel_obj_classes[:num_pairs]

                # 处理 rel_importance，使其大小与 num_pairs 一致
                rel_importance = rel_importance[rel_importance > 0]
                if len(rel_importance) < num_pairs:
                    diff = num_pairs - len(rel_importance)
                    rel_importance = torch.cat([rel_importance, torch.full((diff,), 0.5, device=rel_importance.device)])
                elif len(rel_importance) > num_pairs:
                    rel_importance = rel_importance[:num_pairs]

                # 确保 rel_importance 的 size(0) 与 num_pairs 一致
                repeat_factor_importance = (num_pairs // rel_importance.size(0)) + 1
                rel_importance = rel_importance.repeat(repeat_factor_importance)[:num_pairs]

                rel_obj_classes_rep = self.norm_pair2(rel_obj_classes)
                rel_obj_classes_rep = rel_obj_classes_rep.unsqueeze(0)
                rel_obj_classes_rep = self.poolformer_attention(rel_obj_classes_rep, rel_importance)

                rel_obj_classes_rep = rel_obj_classes_rep.squeeze(dim=0)  # 去掉多余的维度，如果存在

                distance = compute_euclidean_distance(pair_pred_rep, rel_obj_classes_rep)

                # 对距离矩阵进行排序并选取前k个距离
                sorted_distances, _ = torch.sort(distance)
                # Step 3: 归一化距离
                mean_distance = torch.mean(sorted_distances)
                std_distance = torch.std(sorted_distances)
                normalized_distances = (sorted_distances - mean_distance) / std_distance
                trim_ratio = 0.1

                trim_num = int(trim_ratio * len(normalized_distances))
                if trim_num > 0:
                    trimmed_distances = normalized_distances[trim_num:-trim_num]
                else:
                    trimmed_distances = normalized_distances

                dist_loss = torch.mean(torch.abs(trimmed_distances))

                add_losses.update({"rel_im_loss2": dist_loss})
            else:
                add_losses.update({"rel_im_loss2": torch.tensor(0.0).cuda()})
        add_data = {}
        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses, add_data
        else:
            return obj_dists, rel_dists, add_losses, add_data
