import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist


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

    def __init__(self, margin=0.3, normalize_feature=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        if margin > 0 :
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
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

        # For each anchor, find the hardest positive and negative
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


class DivTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, normalize_feature=True):
        super(DivTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        if margin > 0 :
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
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

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        return torch.mean((dist_ap + self.margin) / (dist_an + 1e-5))


class ProbTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, normalize_feature=True):
        super(ProbTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        if margin > 0 :
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, probs):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        probs = probs.detach()
        if self.normalize_feature:
            inputs = normalize(inputs, axis=-1)
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        pos_mask = torch.logical_xor(mask, torch.eye(n, device=mask.device))
        neg_mask = torch.logical_not(mask)

        dist_ap, dist_an = [], []
        for i in range(n):
            tprob = probs[:, targets[i]]
            pos_prob = torch.where(pos_mask[i], tprob, torch.ones_like(tprob, device=mask.device))
            neg_prob = torch.where(neg_mask[i], tprob, torch.zeros_like(tprob, device=mask.device))
            dist_ap.append(dist[i][torch.argmin(pos_prob)].unsqueeze(0))
            dist_an.append(dist[i][torch.argmax(neg_prob)].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        if self.margin > 0:
            return self.ranking_loss(dist_an, dist_ap, y)
        else:
            return self.ranking_loss(dist_an - dist_ap, y)

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
                loss += self.ranking_loss(pos + self.margin - neg)  #self.ranking_loss(neg, pos, y)
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
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.register_parameter('centers', nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda()))
        # self.centers =  nn.Parameter(torch.randn(self.num_classes, self.feat_dim)).cuda()

        nn.init.normal_(self.centers, 0, 0.01)
    def forward(self, x, labels):


        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

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
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=True):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

    # def forward(self, input, label):
    #     # --------------------------- cos(theta) & phi(theta) ---------------------------
    #     #print(input.size(), self.weight.size())
    #     cosine = F.linear(F.normalize(input), F.normalize(self.weight))
    #     sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
    #     phi = cosine * self.cos_m - sine * self.sin_m
    #     if self.easy_margin:
    #         phi = torch.where(cosine > 0, phi, cosine)
    #     else:
    #         phi = torch.where(cosine > self.th, phi, cosine - self.mm)
    #     # --------------------------- convert label to one-hot ---------------------------
    #     # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
    #     one_hot = torch.zeros(cosine.size(), device='cuda')
    #     one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    #     # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    #     output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
    #     output *= self.s
    #     # print(output)
    #
    #     return output

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        #print(input.size(), self.weight.size())
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        angles = torch.acos(torch.clamp(cosine, -1, 1))
        phi = torch.cos(angles + self.m)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
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
    def __init__(self, num_classes, in_feat, scale=64, margin=0.35): # 128 0.15
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


class CamTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, num_class=None, normalize_feature=True):
        super(CamTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.num_class = num_class
        if margin > 0 :
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, cams):
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
        # new_targets = targets + cams * self.num_class
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        cam_mask = cams.expand(n, n).eq(cams.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            mask_ap = cam_mask[i] & (mask[i] == 0)
            mask_an = (cam_mask[i] == 0) & (mask[i] == 0)
            if dist[i][mask_ap].size(0) > 0:
                ap = dist[i][mask_ap].min().unsqueeze(0)
                # print('ap', ap.size())
            else:
                ap = torch.tensor([self.margin + 1], device=mask.device)
            if dist[i][mask_an].size(0) > 0:
                an = dist[i][mask_an].min().unsqueeze(0)
                # print('an', an.size())
            else:
                an = torch.tensor([0.], device=mask.device)
            dist_ap.append(ap)
            dist_an.append(an)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        if self.margin > 0:
            return self.ranking_loss(dist_ap, dist_an, y)
        else:
            return self.ranking_loss(dist_ap - dist_an, y)


def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (2 * Q.shape[1])

        curr_sum = torch.sum(Q, dim=1)
        # dist.all_reduce(curr_sum)

        for it in range(nmb_iters):
            u = curr_sum
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            curr_sum = torch.sum(Q, dim=1)
            # dist.all_reduce(curr_sum)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5 # 0.5
        self.margin = 0.1

        self.scale_pos = 2. #default 2
        self.scale_neg = 40. # default 40
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