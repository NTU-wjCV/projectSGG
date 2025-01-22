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

@registry.ROI_RELATION_PREDICTOR.register("PENET_im")
class PENET_im(nn.Module):
    def __init__(self, config, in_channels):
        super(PENET_im, self).__init__()
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
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)

        self.embed_dim = 300  # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2  # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT

        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR,
                                          wv_dim=self.embed_dim)  # load Glove for objects
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

        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)

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
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

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
        # 添加注意力层
        self.poolformer_attention = SequencePoolFormer(embed_dim=self.mlp_dim // 2, depth=2)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, rel_importance,
                rel_obj_classes
                , logger=None):

        add_losses = {}
        add_data = {}

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        #####

        entity_rep = self.post_emb(roi_features)  # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)  # xo

        entity_embeds = self.obj_embed(entity_preds)  # obtaining the word embedding of entities with GloVe

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

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

        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        ######

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if rel_labels is None:
            rel_labels = [torch.tensor([]).cuda()]  # 假设你在使用GPU，如果使用CPU，去掉 .cuda()
        else:
            rel_labels = [torch.tensor(label).cuda() for label in rel_labels] # 假设 rel_labels 已经是一个列表，并且你在使用GPU

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
            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach()
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51 * 51)
            ### end

            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, 51, -1)
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(51, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(
                dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1  # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(51).cuda(),
                                  -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})
            ### end

            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 51, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(
                dim=2) ** 2  # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), 51).cuda()
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(
                dim=1) / 10  # obtaining g-, where k1 = 10,
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(),
                                 distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})  # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end

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