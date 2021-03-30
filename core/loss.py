import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, normalize_feature=True, reduce='mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        if margin > 0 :
            self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction=reduce)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def compute_distance(self, inputs, **kwargs):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        return dist

    def compute_loss(self, dist, targets, **kwargs):
        n = dist.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        if self.margin > 0:
            return self.ranking_loss(dist_an, dist_ap, y)
        else:
            return self.ranking_loss(dist_an - dist_ap, y)

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        if self.normalize_feature:
            inputs = normalize(inputs, axis=-1)

        dist = self.compute_distance(inputs)
        # For each anchor, find the hardest positive and negative
        return self.compute_loss(dist, targets)


class VGTripletLoss(TripletLoss):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, normalize_feature=True):
        super(VGTripletLoss, self).__init__(margin, normalize_feature)

    def compute_distance(self, inputs, **kwargs):
        global_inputs, partial_input = inputs  # global_inputs b * 512 partial_input b * 6 * 256
        n = global_inputs.size(0)
        # calc the distance of pg_global
        # dist = torch.zeros((n, n), device=global_inputs.device)
        if self.normalize_feature:
            global_inputs = F.normalize(global_inputs, p=2, dim=-1)  # global_inputs b * 512

        if self.normalize_feature:
            partial_input = F.normalize(partial_input, p=2, dim=-1)  # partial_input b * 6 * 256

        pl = kwargs.get('part_labels')  # b * 6

        gs = torch.matmul(global_inputs, global_inputs.t())
        gs = (1 - gs) / 2

        pl_0, pl_1 = pl.unsqueeze(1), pl.unsqueeze(0)
        overlap = pl_0 * pl_1  # b * b * 6

        slf = (1. - torch.matmul(partial_input.permute(1, 0, 2), partial_input.permute(1, 2, 0))) / 2  # 6 * N * N
        slf = slf.permute(1, 2, 0) * overlap  # N * N * 6

        dist = (slf.sum(-1) + gs) / (overlap.sum(-1) + 1)

        # for i in range(n):
        #     # calc the distance of pg_global
        #     gf = global_inputs[i]
        #     gf = gf.expand_as(global_inputs)
        #     gs = gf * global_inputs
        #     gs = gs.sum(1)
        #     gs = (gs + 1.) / 2  # [batchsize]
        #     # Calculate the distance of partial features
        #     lpl = pl[i]
        #     overlap = (pl * lpl).float()
        #     overlap = overlap.view(-1, pl.size(1))
        #
        #     pf = partial_input[i]
        #     pf = pf.expand_as(partial_input)
        #     ps = pf * partial_input
        #     ps = ps.sum(2)
        #     ps = (ps + 1.) / 2
        #     ps = ps * overlap
        #
        #     s = (ps.sum(1) + gs) / (overlap.sum(1) + 1)
        #     dist[i] = s

        return dist

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        global_labels, part_labels = targets
        dist = self.compute_distance(inputs, part_labels=part_labels)

        return self.compute_loss(dist, global_labels)

class PairTripletLoss(TripletLoss):


    def __init__(self, margin=0.3, normalize_feature=True):
        super(PairTripletLoss, self).__init__(margin, normalize_feature)

    def forward(self, input_0, input_1, targets):
        if self.normalize_feature:
            input_0 = normalize(input_0, axis=-1)
            input_1 = normalize(input_1, axis=-1)
        n = targets.size(0)
        pairwise_dist = torch.sqrt(torch.sum(torch.square(input_0 - input_1), dim=-1) + 1e-12)
        pairwise_dist = torch.reshape(pairwise_dist, (n, n))

        return self.compute_loss(pairwise_dist, targets)


class SoftTripletLoss(nn.Module):


    def __init__(self, margin=0.3, normalize_feature=True, scale=1.):
        super(SoftTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.scale = scale
        if margin > 0:
            self.ranking_loss = nn.ReLU(inplace=True)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        if self.normalize_feature:
            inputs = normalize(inputs, axis=-1)
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        dist_s = dist * self.scale
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        pos_mask = torch.logical_xor(mask, torch.eye(n, dtype=torch.bool, device=mask.device))
        loss = 0
        for i in range(n):
            neg = dist_s[i][mask[i] == 0]
            neg_weight = (-neg).exp() / (-neg).exp().sum()
            neg = (neg*neg_weight).sum()

            pos = dist_s[i][pos_mask[i]]
            pos_weight = pos.exp() / pos.exp().sum()
            pos = (pos * pos_weight).sum()

            # Compute ranking hinge loss
            y = torch.ones_like(pos)
            if self.margin > 0:
                loss += self.ranking_loss(pos + self.margin - neg)
            else:
                loss += self.ranking_loss(neg - pos, y)
        return loss / n



class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = torch.tensor(math.cos(m))
        self.sin_m = torch.tensor(math.sin(m))
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        #print(input.size(), self.weight.size())
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # angles = torch.acos(torch.clamp(cosine, -1, 1))
        # phi = torch.cos(angles + self.m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output)

        return output


def one_hot(labels, num_classes, dtype=None):
    # eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    device = labels.device
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)


class Circle(nn.Module):


    def __init__(self, num_classes, in_feat, scale=64, margin=0.35):
        super().__init__()
        self._num_classes = num_classes
        self._s = scale
        self._m = margin

        self.weight = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features, targets):
        sim_mat = F.linear(F.normalize(features), F.normalize(self.weight))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        targets = one_hot(targets, self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        return pred_class_logits


class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.
        self.scale_neg = 40.
        print(self.scale_pos, self.scale_neg)

    def forward(self, feats, labels):
        feats = normalize(feats, axis=-1)
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]
            # print(max(pos_pair_),min(pos_pair_), max(neg_pair_), min(neg_pair_))
            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss


class AdapativeCrossEntropyLabelSmooth(CrossEntropyLabelSmooth):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(AdapativeCrossEntropyLabelSmooth, self).__init__(num_classes, epsilon, use_gpu)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        pt = torch.softmax(inputs, dim=1)[range(inputs.size(0)), targets] # b
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        epsilon = (self.epsilon * (1 - pt)).unsqueeze(dim=1).detach()
        targets = (1 - epsilon) * targets + epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class LSRWithDirection(CrossEntropyLabelSmooth):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.2, use_gpu=True, adaptive=True):
        super(LSRWithDirection, self).__init__(num_classes, epsilon, use_gpu)
        self.adaptive = adaptive

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        n, c = inputs.size()
        assert c % self.num_classes == 0
        log_probs = self.logsoftmax(inputs)

        pt = torch.softmax(inputs, dim=1)  # b
        direct_pt = pt.reshape((n, c // 3, 3))
        class_pt = direct_pt.sum(dim=-1)

        id_targets, direct_targets = targets
        final_targets = id_targets * 3 + direct_targets

        new_targets = torch.zeros_like(log_probs).scatter_(1, final_targets.unsqueeze(1), 1)
        new_targets_2 = torch.zeros((n, c // 3), device=new_targets.device).scatter_(1, id_targets.unsqueeze(1), 1)
        new_targets_2 = new_targets_2.unsqueeze(-1).repeat((1, 1, 3)).reshape((n, c))

        if self.adaptive:
            epsilon_1 = (self.epsilon * (1 - class_pt[range(n), id_targets])).unsqueeze(dim=1).detach()
            epsilon_2 = (self.epsilon * (1 - pt[range(n), final_targets])).unsqueeze(dim=1).detach()
        else:
            epsilon_1 = self.epsilon
            epsilon_2 = self.epsilon

        new_targets = (1 - epsilon_1 - epsilon_2) * new_targets + epsilon_2 * (1 - new_targets_2) / (c - 3) + \
                  epsilon_1 * (new_targets_2 - new_targets) / 2
        loss = (- new_targets * log_probs).mean(0).sum()
        return loss

