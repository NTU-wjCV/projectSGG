import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .model_gpsnet import Boxes_Encode, Boxes_su_Encode, Boxes_uo_Encode
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info

from .utils_motifs import to_onehot, encode_box_info
from maskrcnn_benchmark.modeling.make_layers import make_fc
from timm.models.layers import DropPath, trunc_normal_

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
        self.spt_dim = 64
        self.spt_su_dim = 32
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

        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
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

        # 111
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
        # self.poolformer_attention = SequencePoolFormer(embed_dim=self.mlp_dim // 2, depth=2)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, rel_importance,
                logger=None):

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

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)
            #
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo
            sub = s_embed  # s = Ws x ts
            obj = o_embed  # o = Wo x to
            ##### for the model convergence
            sub = self.norm_sub(torch.relu(self.linear_sub(sub)))
            obj = self.norm_obj(torch.relu(self.linear_obj(obj)))
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj))  # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp
        rel_rep = fusion_so  # F(s,o)
        rel_rep = fusion_so - sem_pred * gate_sem_pred  # F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        # self.rel_embed.weight 提取嵌入层的权重，这是一个大小为[self.num_rel_cls, self.embed_dim]的矩阵，其中每一行是一个关系的嵌入向量。
        ##### for the model convergence 加速模型的收敛，提升模型的泛化能力，并防止过拟合
        rel_rep = self.norm_rel_rep(torch.relu(self.linear_rel_rep(rel_rep)))
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        # project_head是MLP
        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        #####
        rel_dists = rel_rep.split(num_rels, dim=0)
        # norm 函数用于计算张量的范数。dim 参数指定了沿着哪个维度计算范数，而 keepdim 参数则指定是否保持原有的维度。
        # 当 keepdim=True 时，计算后的张量仍然保持原有的维度，其它维度的大小将变为1。
        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        entity_dists = entity_dists.split(num_objs, dim=0)

        rel_dists = rel_dists.split(num_rels, dim=0)  # 训练时，这里的rel_dist应该是每个不同实例之间都会形成一个预测。

        if self.training:
            ### Prototype Regularization  ---- cosine similarity
            # clone()创建一个与predicate_proto_norm具有相同值但独立的新Tensor，detach()方法用于断开梯度，
            # 即返回一个新的Tensor，该Tensor与predicate_proto_norm共享存储空间，但是梯度不会在该Tensor上累积。这里目的是为了在训练时不对predicate_proto_norm的梯度造成影响。
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach()
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51 * 51)  # 51为谓词原型的数量
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
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).abs().sum(
                dim=2)  # Distance Matrix D, dij = ||ci - cj||_1
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1  # obtain d-, where k2 = 1
            # dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_man = max(0, -(d-) + gamma2) # 原
            dist_loss = torch.clamp(-topK_proto_dis + gamma2, min=0).mean()  # 使用 .clamp() 函数将负的距离设为0，然后求平均值得到损失 1

            add_losses.update({"dist_loss2": dist_loss})
            ### end

            ###  Prototype-based Learning  ---- Manhattan Distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 51, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci

            predicate_proto_expand = self.poolformer_attention(predicate_proto_expand, rel_importance)

            # distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            distance_set = (rel_rep_expand - predicate_proto_expand).abs().sum(dim=2)  # 曼哈顿距离计算方式修改为绝对值之和
            mask_neg = torch.ones(rel_labels.size(0), 51).cuda()
            mask_neg[torch.arange(rel_labels.size(
                0)), rel_labels] = 0  # 将每一行的第rel_labels[i]个元素变成0。因此，mask_neg的每一行中，只有一个元素是0，其他元素是1。这个操作可以理解为，对于每一个正样本，将它所对应的那一列的所有元素都变成0。
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(
                dim=1) / 10  # obtaining g-, where k1 = 10,
            # loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean() # 原
            loss_sum = torch.clamp(distance_set_pos - topK_sorted_distance_set_neg + gamma1,
                                   min=0).mean()  # torch.clamp() 将负向差距限制为最小值为0 1

            add_losses.update({"loss_dis": loss_sum})  # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end

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
        obj_dists = obj_dists.split(num_objs, dim=0)  # obj_dists 根据每张图像中目标的数量 num_objs 进行切分，每张图像对应一个概率分布。
        obj_preds = []
        for i in range(len(num_objs)):
            # 计算候选框之间的 IoU 重叠度矩阵，并判断哪些候选框重叠度大于阈值 self.nms_thresh。
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= 0.7  # self.nms_thresh # (#box, #box, #class)
            # 对目标概率分布进行 softmax 处理，得到每个目标在每个类别上的概率分布。
            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1  # 将背景类别的概率分布设置为负数，这样在后续的处理中，背景类别不会被选为最终的目标预测结果
            # 初始化每张图像的目标预测标签为 0
            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                # np.unravel_index函数用于将一个一维索引转换为多维索引，返回一个元组，包含最大值在out_dists_sampled中对应的行和列索引
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[
                    is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0  # 将与当前检测框 box_ind 重叠度大于阈值的检测框在当前目标类别上的概率分布设置为 0。
                out_dists_sampled[
                    box_ind] = -1.0  # This way we won't re-sample 将当前检测框的概率分布设置为负数，这样在后续的处理中，该检测框不会被选为最终的目标预测结果

            obj_preds.append(out_label.long())  # 将当前图像的目标预测结果加入到结果列表中
        obj_preds = torch.cat(obj_preds, dim=0)  # 将所有图像的目标预测结果拼接成一个张量
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