# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
from torch.nn import functional as F

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

        # self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        # self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        # self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        # self.vis2sem = nn.Sequential(*[
        #     nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
        #     nn.Dropout(dropout_p), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        # ])
        #
        # self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim // 2, 2)

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

        # self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)
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
        if rel_obj_classes is not None:
            # 合并 rel_obj_classes
            rel_obj_classes_combined = torch.cat(rel_obj_classes, dim=0)

            # 找到 rel_importance 中不为 0 的值
            non_zero_importance = rel_importance[rel_importance > 0]

            # 检查长度，如果不一致，扩充 non_zero_importance
            if len(rel_obj_classes_combined) != len(non_zero_importance):
                diff = abs(len(rel_obj_classes_combined) - len(non_zero_importance))
                non_zero_importance = torch.cat(
                    [non_zero_importance, torch.full((diff,), 0.5, device=non_zero_importance.device)])

            # 将 rel_obj_classes 和 non_zero_importance 组合成一个列表
            obj_importance_pairs = list(zip(rel_obj_classes_combined.tolist(), non_zero_importance.tolist()))

            # 计算扩充数量
            num_to_expand = pair_pred.shape[0] - rel_obj_classes_combined.shape[0]

            # 扩充 rel_obj_classes_combined，并打乱顺序
            expanded_rel_obj_classes = rel_obj_classes_combined[
                torch.randint(0, rel_obj_classes_combined.shape[0], (num_to_expand,))]
            expanded_rel_obj_classes_combined = torch.cat([rel_obj_classes_combined, expanded_rel_obj_classes],
                                                              dim=0)
            shuffled_indices = torch.randperm(expanded_rel_obj_classes_combined.shape[0])
            expanded_rel_obj_classes_combined = expanded_rel_obj_classes_combined[shuffled_indices]

            # 创建新的 rel_importance
            new_rel_importance = torch.zeros_like(rel_importance)

            # 填充新的 rel_importance
            index = 0
            for obj_class in expanded_rel_obj_classes_combined.tolist():
                for pair, importance in obj_importance_pairs:
                    if pair == obj_class:
                        new_rel_importance[index] = importance
                        index += 1
                        break

            # 更新 rel_importance
            rel_importance = new_rel_importance
            # 打乱顺序但保持组合不变
            shuffled_indices = torch.randperm(pair_pred.shape[0])
            shuffled_pair_pred = pair_pred[shuffled_indices]
            pair_pred = shuffled_pair_pred.float()
            rel_obj_classes = expanded_rel_obj_classes_combined


        rel_rep = fusion_so  # F(s,o)
        # self.rel_embed.weight 提取嵌入层的权重，这是一个大小为[self.num_rel_cls, self.embed_dim]的矩阵，其中每一行是一个关系的嵌入向量。
        ##### for the model convergence 加速模型的收敛，提升模型的泛化能力，并防止过拟合
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm

        predicate_proto = self.W_pred(self.rel_embed.weight)
        predicate_proto = self.norm_pred(self.dropout_pred(torch.relu(self.linear_pred(predicate_proto))))

        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        rel_dists = rel_dists.split(num_rels, dim=0)
        ######
        entity_dists = entity_dists.split(num_objs, dim=0)

        if self.training:
            def compute_euclidean_distance(x1, x2):
                return (x1 - x2).norm(dim=2) ** 2
            if rel_obj_classes is not None:
                pair_pred_rep = self.norm_pair1(pair_pred.float())
                rel_obj_classes_rep = self.norm_pair2(rel_obj_classes)
                rel_obj_classes_rep = rel_obj_classes_rep.unsqueeze(0)
                rel_obj_classes_rep = self.poolformer_attention(rel_obj_classes_rep, rel_importance)
                # 获取动态长度
                num_pairs = pair_pred_rep.size(0)

                rel_obj_classes_rep = rel_obj_classes_rep.squeeze(dim=0)  # 去掉多余的维度，如果存在
                # 计算距离矩阵
                pair_pred_rep_a = pair_pred_rep.unsqueeze(dim=1).expand(-1, num_pairs, -1)
                rel_obj_classes_rep_b = rel_obj_classes_rep.unsqueeze(dim=0).expand(num_pairs, -1, -1)
                distance_matrix = compute_euclidean_distance(pair_pred_rep_a, rel_obj_classes_rep_b)

                # 对距离矩阵进行排序并选取前k个距离
                sorted_distance_matrix, _ = torch.sort(distance_matrix, dim=1)
                topK_distance = sorted_distance_matrix[:, :2].sum(dim=1) / 1  # 取前2个距离的平均值

                # 计算损失
                gamma2 = 5.0
                dist_loss = torch.clamp(-topK_distance + gamma2, min=0).mean()
                add_losses.update({"dist_loss2": dist_loss})
                # ´òÓ¡ rel_importance µÄÖµ
                print(f"rel_importance: {rel_importance}")

                # ´òÓ¡ pair_pred_rep ºÍ rel_obj_classes_rep µÄÐÎ×´ºÍ²¿·ÖÖµ
                print(f"pair_pred_rep shape: {pair_pred_rep.shape}")
                print(f"pair_pred_rep sample: {pair_pred_rep[:2]}")

                print(f"rel_obj_classes_rep shape: {rel_obj_classes_rep.shape}")
                print(f"rel_obj_classes_rep sample: {rel_obj_classes_rep[:2]}")

                # ´òÓ¡¾àÀë¾ØÕó
                distance_matrix = compute_euclidean_distance(pair_pred_rep_a, rel_obj_classes_rep_b) / 1000.0
                print(f"distance_matrix sample: {distance_matrix[:2, :2]}")

                # ´òÓ¡ÅÅÐòºóµÄ¾àÀë¾ØÕó
                sorted_distance_matrix, _ = torch.sort(distance_matrix, dim=1)
                print(f"sorted_distance_matrix sample: {sorted_distance_matrix[:2, :2]}")

                # ´òÓ¡ topK_distance
                topK_distance = sorted_distance_matrix[:, :2].sum(dim=1) / 1
                print(f"topK_distance: {topK_distance}")

                # ´òÓ¡¼ÆËãµÄËðÊ§
                gamma2 = 5.0
                dist_loss = torch.clamp(-topK_distance + gamma2, min=0).mean()
                print(f"dist_loss: {dist_loss}")
            else:
                add_losses.update({"dist_loss2": torch.tensor(0.0).cuda()})

        return entity_dists, rel_dists, add_losses, add_data

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:  # 不使用目标标签，后两个指标任务中
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals],
                             dim=0).detach()  # .detach() 是 PyTorch 中的一个方法，它会返回一个新的 Tensor，这个 Tensor 不会跟踪梯度信息。
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))  # 这个应该是盒子的位置嵌入
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))  # 全连接层

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels  # 如果使用了标注的盒子和标签，则目标预测结果就是标签
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)  # 目标的概率分布就是对标签进行热编码
        else:  # 没有使用目标标签
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151 ，一个全连接层，
            use_decoder_nms = self.mode == 'sgdet' and not self.training  # 指标为使用原图，没有盒子和标签，并且不在训练的时候，则使用非极大值抑制，但是这个decoder是什么？
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]  # 盒子分类从proposal提
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()  # 目标预测
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()

        return obj_dists, obj_preds
    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0) # obj_dists 根据每张图像中目标的数量 num_objs 进行切分，每张图像对应一个概率分布。
        obj_preds = []
        for i in range(len(num_objs)):
            # 计算候选框之间的 IoU 重叠度矩阵，并判断哪些候选框重叠度大于阈值 self.nms_thresh。
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >=0.7 # self.nms_thresh # (#box, #box, #class)
            # 对目标概率分布进行 softmax 处理，得到每个目标在每个类别上的概率分布。
            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1 # 将背景类别的概率分布设置为负数，这样在后续的处理中，背景类别不会被选为最终的目标预测结果
            # 初始化每张图像的目标预测标签为 0
            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                # np.unravel_index函数用于将一个一维索引转换为多维索引，返回一个元组，包含最大值在out_dists_sampled中对应的行和列索引
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0 # 将与当前检测框 box_ind 重叠度大于阈值的检测框在当前目标类别上的概率分布设置为 0。
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample 将当前检测框的概率分布设置为负数，这样在后续的处理中，该检测框不会被选为最终的目标预测结果

            obj_preds.append(out_label.long()) # 将当前图像的目标预测结果加入到结果列表中
        obj_preds = torch.cat(obj_preds, dim=0) # 将所有图像的目标预测结果拼接成一个张量
        return obj_preds



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
@registry.ROI_RELATION_PREDICTOR.register("PrototypeEmbeddingNetwork")
class PrototypeEmbeddingNetwork(nn.Module):
    def __init__(self, config, in_channels):
        super(PrototypeEmbeddingNetwork, self).__init__()
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

        self.mlp_dim = 2048 # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.spt_dim = 64
        self.spt_su_dim = 32
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)

        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2 # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT
        
        # 预训练的object的word embedding矩阵
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        # 预训练的relation的word embedding矩阵
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
       
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim*2, self.mlp_dim)  
        self.gate_obj = nn.Linear(self.mlp_dim*2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim*2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim // 2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
       
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)
        # nn.Parameter 是一个 Tensor，但它会被自动添加到 Parameter List 中，当在使用 nn.Module 定义神经网络时，所有 Parameter 都会被自动注册到模型的参数中，并用于自动求导和梯度更新。
        # 也就是说，当我们需要在训练过程中对某些 Tensor 的值进行优化时，我们可以使用 nn.Parameter 定义这些 Tensor。
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim) # 创建一个输入维度为 obj_dim + embed_dim + 128，输出维度为hidden_dim 的全连接层

        #111
        self.fusion = nn.Sequential(
            nn.Linear(self.mlp_dim + self.spt_dim, self.mlp_dim // 2),
            nn.ReLU(True), nn.Dropout(dropout_p),
            nn.Linear(self.mlp_dim // 2, self.mlp_dim),
        )
        self.fusion_su = nn.Sequential(
            nn.Linear(self.mlp_dim + self.spt_su_dim, self.mlp_dim // 2),
            nn.ReLU(True), nn.Dropout(dropout_p),
            nn.Linear(self.mlp_dim // 2, self.mlp_dim),
        )


        self.norm_fusion = nn.LayerNorm(self.mlp_dim)

        self.get_boxes_encode = Boxes_Encode()
        self.get_su_boxes_encode = Boxes_su_Encode()
        self.get_uo_boxes_encode = Boxes_uo_Encode()


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

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, rel_importance, logger=None):

        # 检查 rel_labels 和 rel_importance 是否为 None
        if rel_labels is None or rel_importance is None:
            rel_labels = [torch.tensor([]).cuda()]  # 假设你在使用GPU，如果使用CPU，去掉 .cuda()
            rel_importance = torch.zeros(len(rel_labels)).cuda()
        else:
            if isinstance(rel_importance, list):
                rel_importance = [torch.tensor(ri) for ri in rel_importance]
                rel_importance = torch.cat(rel_importance).cuda()  # 假设你在使用GPU，如果使用CPU，去掉 .cuda()

        add_losses = {}
        add_data = {}

        # refine object labels 细化目标标签
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals) # 目标实体的概率分布和预测结果
        ##### 

        entity_rep = self.post_emb(roi_features)   # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)    # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)    # xo
        # 实体嵌入是由实体的预测结果进行词嵌入过来的
        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 使用GloVe获取实体嵌入

        num_rels = [r.shape[0] for r in rel_pair_idxs] # 获取关系对的数量列表
        num_objs = [len(b) for b in proposals] # 获取每个proposal中物体的数量
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0) # 将主体实体的表示拆分为proposal中物体数量的列表。
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0) # 将实体的预测值拆分为proposal中物体数量的列表。
        entity_embeds = entity_embeds.split(num_objs, dim=0) # 将实体的嵌入拆分为proposal中物体数量的列表。

        fusion_so = []
        fusion_su = []
        fusion_uo = []
        pair_preds = []
        spt_feats = [] # 111
        spt_su_feats = []  # 111
        spt_uo_feats = []  # 111
        sub_list = []
        obj_list = []

        # for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal, roi_feat in zip(rel_pair_idxs, sub_reps, obj_reps, entity_preds, entity_embeds, proposals, roi_features): # 111
        #     # 111
        #     # print("pair_idx的值", pair_idx)
        #     # print("pair_idx的形状", pair_idx.shape)  # shape属性显示张量形状
        #
        #     if torch.numel(pair_idx) == 0:
        #         if logger is not None:
        #             logger.warning('image {} rel pair idx is emtpy!\nrel_pair_idx:{}\nbboxes:{}'.format(
        #                 proposal.image_fn, str(pair_idx), str(proposal)))
        #         spt_feats.append(torch.empty((0, 64)).to(roi_feat))
        #         spt_su_feats.append(torch.empty((0, 32)).to(roi_feat))
        #         continue
        #     w, h = proposal.size
        #     bboxes_tensor = proposal.bbox
        #
        #     transfered_boxes = torch.stack(
        #         (
        #             bboxes_tensor[:, 0] / w,
        #             bboxes_tensor[:, 3] / h,
        #             bboxes_tensor[:, 2] / w,
        #             bboxes_tensor[:, 1] / h,
        #             (bboxes_tensor[:, 2] - bboxes_tensor[:, 0]) * \
        #             (bboxes_tensor[:, 3] - bboxes_tensor[:, 1]) / w / h,
        #         ), dim=-1 # 候选框的坐标和面积进行归一化，形状为[N,5]
        #     ).to(roi_feat)
        #     spt_feats.append(self.get_boxes_encode(bboxes_tensor, pair_idx, w, h))  # 空间特征？
        #
        #     spt_su_feats.append(self.get_su_boxes_encode(bboxes_tensor, pair_idx, w, h))
        #     spt_uo_feats.append(self.get_uo_boxes_encode(bboxes_tensor, pair_idx, w, h))
        #
        #
        #     s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts    用了mlp的
        #     o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to
        #
        #     sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs) 视觉转语义  两个线性层
        #     sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)
        #
        #     gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
        #     gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go
        #
        #     sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
        #     obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo
        #
        #     ##### for the model convergence 加速模型的收敛，提升模型的泛化能力，并防止过拟合
        #     sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
        #     obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
        #
        #     sub_list.append(sub)
        #     obj_list.append(obj)
        #     fusion_so.append(fusion_func(sub, obj) ) # F(s, o)
        #     pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1)) # 在第 1 维进行拼接
        #
        # sub_list = cat(sub_list, dim=0)
        # obj_list = cat(obj_list, dim=0)
        #
        # spt_su_feats = cat(spt_su_feats, dim=0)  # 111
        # spt_uo_feats = cat(spt_uo_feats, dim=0)  # 111
        #
        # sub_list = torch.cat([sub_list, spt_su_feats], dim=1)
        # fusion_su = self.fusion_su(sub_list)
        #
        # obj_list = torch.cat([obj_list, spt_uo_feats], dim=1)
        # fusion_uo = self.fusion_su(obj_list)
        # #####
        # fusion_so = fusion_func(fusion_su, fusion_uo)  # F(s, o)
        #
        # pair_pred = cat(pair_preds, dim=0)
        #
        # spt_feats = cat(spt_feats, dim=0) # 111
        #
        # # 将两个张量按照第二维拼接起来
        # fusion_so = torch.cat([fusion_so, spt_feats], dim=1)
        # #使用MLP将第二维转换为2048
        # fusion_so = self.fusion(fusion_so)
        #
        #
        # sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        # gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp
        #
        #
        # rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        #
        # predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps,
                                                                                   entity_preds, entity_embeds,
                                                                                   proposals):
            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  # Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  # Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)

            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj))  # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  # F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        # self.rel_embed.weight 提取嵌入层的权重，这是一个大小为[self.num_rel_cls, self.embed_dim]的矩阵，其中每一行是一个关系的嵌入向量。
        ##### for the model convergence 加速模型的收敛，提升模型的泛化能力，并防止过拟合
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        # project_head是MLP
        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        ######
        # norm 函数用于计算张量的范数。dim 参数指定了沿着哪个维度计算范数，而 keepdim 参数则指定是否保持原有的维度。
        # 当 keepdim=True 时，计算后的张量仍然保持原有的维度，其它维度的大小将变为1。
        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss


        entity_dists = entity_dists.split(num_objs, dim=0)

        rel_dists = rel_dists.split(num_rels, dim=0) # 训练时，这里的rel_dist应该是每个不同实例之间都会形成一个预测。




        if self.training:

            ### Prototype Regularization  ---- cosine similarity
            # clone()创建一个与predicate_proto_norm具有相同值但独立的新Tensor，detach()方法用于断开梯度，
            # 即返回一个新的Tensor，该Tensor与predicate_proto_norm共享存储空间，但是梯度不会在该Tensor上累积。这里目的是为了在训练时不对predicate_proto_norm的梯度造成影响。
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach()
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)   # 51为谓词原型的数量
            add_losses.update({"l21_loss": l21})  # Lr_sim = ||S||_{2,1}
            '''这段代码试图确保不同的谓词原型之间的余弦相似度较小。
            具体来说，它通过最小化L2,1范数来实现这一点。如果不同的谓词原型之间的余弦相似度较大，那么L2,1范数也会较大。
            通过将这个范数添加到总损失中并对其进行最小化，模型被迫使不同的谓词原型更加正交（即它们之间的余弦相似度接近0）。'''

            ### end
            
            ### Prototype Regularization  ---- Manhattan Distance
            gamma2 = 7.0
            delta = 0.5
            # unsqueeze方法将predicate_proto的形状由(51, mlp_dim)变成(1, 51, mlp_dim)，
            # 即在第0个维度上增加一个长度为1的维度，然后expand方法将其沿着第1个维度复制了51遍，最终得到形状为(51, 51, mlp_dim)的predicate_proto_a。
            # 这样做的目的是为了计算predicate_proto两两之间的欧几里得距离。
            # predicate_proto_a是将predicate_proto在第一维扩展为51个，第二维保持不变，第三维保持不变；
            # predicate_proto_b是将predicate_proto在第二维扩展为51个，第一维保持不变，第三维保持不变。
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, 51, -1)
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(51, -1, -1)
            # proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).abs().sum(dim=2)  # Distance Matrix D, dij = ||ci - cj||_1
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1  # obtain d-, where k2 = 1
            # dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_man = max(0, -(d-) + gamma2) # 原
            dist_loss = torch.clamp(-topK_proto_dis + gamma2, min=0).mean() # 使用 .clamp() 函数将负的距离设为0，然后求平均值得到损失 1

            add_losses.update({"dist_loss2": dist_loss})
            ### end 

            ###  Prototype-based Learning  ---- Manhattan Distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 51, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci

            predicate_proto_expand = self.poolformer_attention(predicate_proto_expand, rel_importance)

            #distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            distance_set = (rel_rep_expand - predicate_proto_expand).abs().sum(dim=2)  # 曼哈顿距离计算方式修改为绝对值之和
            mask_neg = torch.ones(rel_labels.size(0), 51).cuda()  
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0 # 将每一行的第rel_labels[i]个元素变成0。因此，mask_neg的每一行中，只有一个元素是0，其他元素是1。这个操作可以理解为，对于每一个正样本，将它所对应的那一列的所有元素都变成0。
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            # loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean() # 原
            loss_sum = torch.clamp(distance_set_pos - topK_sorted_distance_set_neg + gamma1, min=0).mean() # torch.clamp() 将负向差距限制为最小值为0 1

            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end

            # ### 对比损失
            # important_labels = (rel_importance > 0.5).float()
            # for i in range(rel_rep.size(0)):
            #     for j in range(i + 1, rel_rep.size(0)):
            #         output1 = rel_rep[i]
            #         output2 = rel_rep[j]
            #         label = (important_labels[i] == important_labels[j]).float()
            #         contrastive_loss = self.contrastive_loss(output1.unsqueeze(0), output2.unsqueeze(0), label)
            #         if "contrastive_loss" in add_losses:
            #             add_losses["contrastive_loss"] += contrastive_loss
            #         else:
            #             add_losses["contrastive_loss"] = contrastive_loss
            # ### end

        return entity_dists, rel_dists, add_losses, add_data

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))
        # for proposal in proposals:
        #     for field in proposal.fields():
        #         print(field, ":", proposal.get_field(field))
        #         print("形状", proposal.get_field(field).shape)
        #         print('*' * 100)
        #     print(proposal.bbox)
        #     print("形状", proposal.bbox.shape)
        #
        #     print('*' * 200)

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else: # 不使用目标标签，后两个指标任务中
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach() # .detach() 是 PyTorch 中的一个方法，它会返回一个新的 Tensor，这个 Tensor 不会跟踪梯度信息。
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals)) # 这个应该是盒子的位置嵌入
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1)) # 全连接层

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels # 如果使用了标注的盒子和标签，则目标预测结果就是标签
            obj_dists = to_onehot(obj_preds, self.num_obj_classes) # 目标的概率分布就是对标签进行热编码
        else: # 没有使用目标标签
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151 ，一个全连接层，
            use_decoder_nms = self.mode == 'sgdet' and not self.training # 指标为使用原图，没有盒子和标签，并且不在训练的时候，则使用非极大值抑制，但是这个decoder是什么？
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals] # 盒子分类从proposal提
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long() # 目标预测
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0) # obj_dists 根据每张图像中目标的数量 num_objs 进行切分，每张图像对应一个概率分布。
        obj_preds = []
        for i in range(len(num_objs)):
            # 计算候选框之间的 IoU 重叠度矩阵，并判断哪些候选框重叠度大于阈值 self.nms_thresh。
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >=0.7 # self.nms_thresh # (#box, #box, #class)
            # 对目标概率分布进行 softmax 处理，得到每个目标在每个类别上的概率分布。
            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1 # 将背景类别的概率分布设置为负数，这样在后续的处理中，背景类别不会被选为最终的目标预测结果
            # 初始化每张图像的目标预测标签为 0
            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                # np.unravel_index函数用于将一个一维索引转换为多维索引，返回一个元组，包含最大值在out_dists_sampled中对应的行和列索引
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0 # 将与当前检测框 box_ind 重叠度大于阈值的检测框在当前目标类别上的概率分布设置为 0。
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample 将当前检测框的概率分布设置为负数，这样在后续的处理中，该检测框不会被选为最终的目标预测结果

            obj_preds.append(out_label.long()) # 将当前图像的目标预测结果加入到结果列表中
        obj_preds = torch.cat(obj_preds, dim=0) # 将所有图像的目标预测结果拼接成一个张量
        return obj_preds

    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x
    
    
def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2



@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

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
        
        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
                
        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
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
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
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

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
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
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
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

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
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
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True),])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)
        
        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)
        
        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

        
    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append( head_rep[pair_idx[:,0]] - tail_rep[pair_idx[:,1]] )
            else:
                ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list
        
        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats
        
        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep  
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':   # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE': # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            #union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            #union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            #union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            #union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest
            
        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
