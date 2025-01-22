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

@registry.ROI_RELATION_PREDICTOR.register("myNetwork")
class myNetwork(nn.Module):
    def __init__(self, config, in_channels):
        super(myNetwork, self).__init__()
        # 获取配置文件中的参数
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048  # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        # self.spt_dim = 64
        # self.spt_su_dim = 32
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)

        self.embed_dim = 300  # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2  # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT

        # 预训练的object的word embedding矩阵
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR,
                                          wv_dim=self.embed_dim)  # load Glove for objects
        # 预训练的relation的word embedding矩阵
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR,
                                     wv_dim=self.embed_dim)  # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)

        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)
        self.norm_pred = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
        self.dropout_rel = nn.Dropout(dropout_p)

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim // 2, 2)
        # # nn.Parameter 是一个 Tensor，但它会被自动添加到 Parameter List 中，当在使用 nn.Module 定义神经网络时，所有 Parameter 都会被自动注册到模型的参数中，并用于自动求导和梯度更新。
        # # 也就是说，当我们需要在训练过程中对某些 Tensor 的值进行优化时，我们可以使用 nn.Parameter 定义这些 Tensor。
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128,
                                   self.hidden_dim)  # 创建一个输入维度为 obj_dim + embed_dim + 128，输出维度为hidden_dim 的全连接层

        self.norm_fusion = nn.LayerNorm(self.mlp_dim)
        #
        self.get_boxes_encode = Boxes_Encode()
        # self.get_su_boxes_encode = Boxes_su_Encode()
        # self.get_uo_boxes_encode = Boxes_uo_Encode()
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

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # 添加注意力层
        self.poolformer_attention = SequencePoolFormer(embed_dim=self.mlp_dim // 2, depth=2)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, rel_importance, rel_obj_classes
                ,logger=None):

        add_losses = {}
        add_data = {}
        # refine object labels 细化目标标签
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)  # 目标实体的概率分布和预测结果
        #####
        entity_rep = self.post_emb(roi_features)  # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)  # xo
        # 实体嵌入是由实体的预测结果进行词嵌入过来的
        entity_embeds = self.obj_embed(
            entity_preds)  # obtaining the word embedding of entities with GloVe 使用GloVe获取实体嵌入

        num_rels = [r.shape[0] for r in rel_pair_idxs]  # 获取关系对的数量列表
        num_objs = [len(b) for b in proposals]  # 获取每个proposal中物体的数量
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)  # 将主体实体的表示拆分为proposal中物体数量的列表。
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)  # 将实体的预测值拆分为proposal中物体数量的列表。
        entity_embeds = entity_embeds.split(num_objs, dim=0)  # 将实体的嵌入拆分为proposal中物体数量的列表。

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps,
                                                                                   entity_preds, entity_embeds,
                                                                                   proposals):
            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  # Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  # Wo x to

            sub = s_embed  # s = Ws x ts
            obj = o_embed  # o = Wo x to
            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj))  # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))
        fusion_so = cat(fusion_so, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        initial_pair_pred_rep = torch.zeros(pair_pred.size(0), self.mlp_dim, device=pair_pred.device)
        initial_rel_obj_classes_rep = torch.zeros(len(rel_obj_classes), self.mlp_dim, device=pair_pred.device) if rel_obj_classes is not None else None
        pair_pred_rep = initial_pair_pred_rep
        rel_obj_classes_rep = initial_rel_obj_classes_rep

        if rel_labels is None:
            rel_labels = [torch.tensor([]).cuda()]  # 假设你在使用GPU，如果使用CPU，去掉 .cuda()
        else:
            rel_labels = [torch.tensor(label).cuda() for label in rel_labels] # 假设 rel_labels 已经是一个列表，并且你在使用GPU

        if rel_importance is None:
            rel_importance = torch.zeros(len(rel_labels)).cuda()
        else:
            rel_importance = [torch.tensor(ri) for ri in rel_importance]
            rel_importance = torch.cat(rel_importance).cuda()  # 假设你在使用GPU，如果使用CPU，去掉 .cuda()

        rel_rep = fusion_so  # F(s,o)
        # self.rel_embed.weight 提取嵌入层的权重，这是一个大小为[self.num_rel_cls, self.embed_dim]的矩阵，其中每一行是一个关系的嵌入向量。
        ##### for the model convergence 加速模型的收敛，提升模型的泛化能力，并防止过拟合
        rel_rep = self.norm_rel_rep(torch.relu(self.linear_rel_rep(rel_rep)))
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm

        predicate_proto = self.W_pred(self.rel_embed.weight)
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ

        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        rel_dists = rel_dists.split(num_rels, dim=0)
        ######
        entity_dists = entity_dists.split(num_objs, dim=0)

        def compute_euclidean_distance(x1, x2):
            diff = x1 - x2
            dist_sq = torch.sum(diff ** 2, dim=-1)
            dist = torch.sqrt(dist_sq)
            return dist

        if self.training:
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach()
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51 * 51)  # 51为谓词原型的数量
            add_losses.update({"l21_loss": l21})  # Lr_sim = ||S||_{2,1}

            # if rel_obj_classes is not None:
            #     num_pairs = pair_pred.size(0)
            #     pair_pred_rep = self.norm_pair1(pair_pred.float())
            #     rel_obj_classes = torch.cat(rel_obj_classes, dim=0)
            #
            #     # 确保 rel_obj_classes 的 size(0) 与 num_pairs 一致
            #     if rel_obj_classes.size(0) < num_pairs:
            #         repeat_factor = (num_pairs // rel_obj_classes.size(0)) + 1
            #         rel_obj_classes = rel_obj_classes.repeat(repeat_factor, 1)[:num_pairs]
            #     else:
            #         rel_obj_classes = rel_obj_classes[:num_pairs]
            #
            #     # 处理 rel_importance，使其大小与 num_pairs 一致
            #     rel_importance = rel_importance[rel_importance > 0]
            #     if len(rel_importance) < num_pairs:
            #         diff = num_pairs - len(rel_importance)
            #         rel_importance = torch.cat([rel_importance, torch.full((diff,), 0.5, device=rel_importance.device)])
            #     elif len(rel_importance) > num_pairs:
            #         rel_importance = rel_importance[:num_pairs]
            #
            #     # 确保 rel_importance 的 size(0) 与 num_pairs 一致
            #     repeat_factor_importance = (num_pairs // rel_importance.size(0)) + 1
            #     rel_importance = rel_importance.repeat(repeat_factor_importance)[:num_pairs]
            #
            #     rel_obj_classes_rep = self.norm_pair2(rel_obj_classes)
            #     rel_obj_classes_rep = rel_obj_classes_rep.unsqueeze(0)
            #     rel_obj_classes_rep = self.poolformer_attention(rel_obj_classes_rep, rel_importance)
            #
            #     rel_obj_classes_rep = rel_obj_classes_rep.squeeze(dim=0)  # 去掉多余的维度，如果存在
            #
            #     distance = compute_euclidean_distance(pair_pred_rep, rel_obj_classes_rep)
            #
            #     # 对距离矩阵进行排序并选取前k个距离
            #     sorted_distances, _ = torch.sort(distance)
            #     # Step 3: 归一化距离
            #     mean_distance = torch.mean(sorted_distances)
            #     std_distance = torch.std(sorted_distances)
            #     normalized_distances = (sorted_distances - mean_distance) / std_distance
            #     trim_ratio = 0.1
            #
            #     trim_num = int(trim_ratio * len(normalized_distances))
            #     if trim_num > 0:
            #         trimmed_distances = normalized_distances[trim_num:-trim_num]
            #     else:
            #         trimmed_distances = normalized_distances
            #
            #     dist_loss = torch.mean(torch.abs(trimmed_distances))
            #
            #     add_losses.update({"dist_loss2": dist_loss})
            # else:
            #     add_losses.update({"dist_loss2": torch.tensor(0.0).cuda()})

        return entity_dists, rel_dists, add_losses, add_data