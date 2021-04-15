import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from pytorch_transformers import BertModel, BertConfig, BertTokenizer
from core.config import config


def l2norm(X, dim, eps=1e-8):
    """
    L2-normalize columns of X
    """

    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)

    return X


def compute_similarity(x1, x2, dim=1, eps=1e-8):
    """
    Returns cosine similarity between x1 and x2, computed along dim.
    """

    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand, eps=1e-8):
    """
    query: (batch, queryL, d)
    context: (batch, sourceL, d)
    """

    batch_size, queryL, sourceL = txt_i_key_expand.size(
        0), local_img_query.size(1), txt_i_key_expand.size(1)
    local_img_query_norm = l2norm(local_img_query, dim=-1)
    txt_i_key_expand_norm = l2norm(txt_i_key_expand, dim=-1)

    # Step 1: preassign attention
    # --> (batch, d, queryL)
    local_img_queryT = torch.transpose(local_img_query_norm, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    attn = torch.bmm(txt_i_key_expand_norm, local_img_queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * config.get('model_config')['lambda_softmax'])
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # print('attn: ', attn)

    # Step 2: identify irrelevant fragments
    # Learning an indicator function H, one for relevant, zero for irrelevant
    if config.get('model_config')['focal_type'] == 'equal':
        funcH = focal_equal(attn, batch_size, queryL, sourceL)
    elif config.get('model_config')['focal_type'] == 'prob':
        funcH = focal_prob(attn, batch_size, queryL, sourceL)
    else:
        funcH = None
    

    # Step 3: reassign attention
    if funcH is not None:
        tmp_attn = funcH * attn
        attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
        attn = tmp_attn / attn_sum

    # --> (batch, d, sourceL)
    txt_i_valueT = torch.transpose(txt_i_value_expand, 1, 2)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(txt_i_valueT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    #return weightedContext, attn
    return weightedContext

def focal_equal(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """

    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def focal_prob(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as the sqrt
    of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (batch, queryL, sourceL)
    """

    # -> (batch, queryL, sourceL, 1)
    xi = attn.unsqueeze(-1).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj = attn.unsqueeze(2).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj_confi = torch.sqrt(xj)

    xi = xi.view(batch_size * queryL, sourceL, 1)
    xj = xj.view(batch_size * queryL, 1, sourceL)
    xj_confi = xj_confi.view(batch_size * queryL, 1, sourceL)

    # -> (batch*queryL, sourceL, sourceL)
    term1 = torch.bmm(xi, xj_confi)
    term2 = xj * xj_confi
    funcF = torch.sum(term1 - term2, dim=-1)  # -> (batch*queryL, sourceL)
    funcF = funcF.view(batch_size, queryL, sourceL)

    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def compute_weiTexts(local_img_query, local_img_value, local_text_key, local_text_value, text_length):
    """
    Compute weighted text embeddings
    :param image_embeddings: Tensor with dtype torch.float32, [n_img, n_region, d]
    :param text_embeddings: Tensor with dtype torch.float32, [n_txt, n_word, d]
    :param text_length: list, contain length of each sentence, [batch_size]
    :param labels: Tensor with dtype torch.int32, [batch_size]
    :return: i2t_similarities: Tensor, [n_img, n_txt]
              t2i_similarities: Tensor, [n_img, n_txt]
    """

    n_img = local_img_query.shape[0]
    n_txt = local_text_key.shape[0]
    t2i_similarities = []
    i2t_similarities = []
    #atten_final_result = {}
    for i in range(n_txt):
        # Get the i-th text description
        n_word = text_length[i]
        #print(i)
        #print(n_word)
        #print(local_text_key.shape)
        #print(local_text_key[i, :n_word, :].shape)
        txt_i_key = local_text_key[i, :n_word, :].unsqueeze(0).contiguous()
        txt_i_value = local_text_value[i, :n_word, :].unsqueeze(0).contiguous()
           
        # -> (n_img, n_word, d)
        txt_i_key_expand = txt_i_key.repeat(n_img, 1, 1)
        txt_i_value_expand = txt_i_value.repeat(n_img, 1, 1)

        # -> (n_img, n_region, d)
        #weiText, atten_text = func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand, args)
        weiText = func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand)
        #atten_final_result[i] = atten_text[i, :, :]
        # image_embeddings = l2norm(image_embeddings, dim=2)
        weiText = l2norm(weiText, dim=2)
        i2t_sim = compute_similarity(local_img_value, weiText, dim=2)
        i2t_sim = i2t_sim.mean(dim=1, keepdim=True)
        i2t_similarities.append(i2t_sim)

        # -> (n_img, n_word, d)
        #weiImage, atten_image = func_attention_MxN(txt_i_key_expand, local_img_query, local_img_value, args)
        weiImage = func_attention_MxN(txt_i_key_expand, local_img_query, local_img_value)
        # txt_i_expand = l2norm(txt_i_expand, dim=2)
        weiImage = l2norm(weiImage, dim=2)
        t2i_sim = compute_similarity(txt_i_value_expand, weiImage, dim=2)
        t2i_sim = t2i_sim.mean(dim=1, keepdim=True)
        t2i_similarities.append(t2i_sim)

    # (n_img, n_txt)
    #torch.save(atten_final_result, 'atten_final_result.pt')
    i2t_similarities = torch.cat(i2t_similarities, 1)
    t2i_similarities = torch.cat(t2i_similarities, 1)

    return i2t_similarities, t2i_similarities


def compute_topk(query_global, query, value_bank, gallery_global, gallery_key, gallery_value,
                       gallery_length, target_query, target_gallery, k_list=[1, 5, 20], reverse=False):
    global_result = []
    local_result = []
    result = []
    sim_cosine = []

    query_global = F.normalize(query_global, p=2, dim=1)
    gallery_global = F.normalize(gallery_global, p=2, dim=1)

    sim_cosine_global = torch.matmul(query_global, gallery_global.t())

    
    for i in range(0, query.shape[0], 200):
        i2t_sim, t2i_sim = compute_weiTexts(query[i:i + 200], value_bank[i:i + 200], gallery_key, gallery_value, gallery_length)
        sim = i2t_sim
        sim_cosine.append(sim)

    sim_cosine = torch.cat(sim_cosine, dim=0)
    
    sim_cosine_all = sim_cosine_global + sim_cosine
    reid_sim = None
    if(config.get('model_config')['reranking']):
        reid_sim = torch.matmul(query_global, query_global.t())

    global_result.extend(topk(sim_cosine_global, target_gallery, target_query, k=k_list))
    if reverse:
        global_result.extend(topk(sim_cosine_global, target_query, target_gallery, k=k_list, dim=0, print_index=False))

    local_result.extend(topk(sim_cosine, target_gallery, target_query, k=k_list))
    if reverse:
        local_result.extend(topk(sim_cosine, target_query, target_gallery, k=k_list, dim=0, print_index=False))

    result.extend(topk(sim_cosine_all, target_gallery, target_query, k=k_list, reid_sim=reid_sim))
    if reverse:
        result.extend(topk(sim_cosine_all, target_query, target_gallery, k=k_list, dim=0, print_index=False, reid_sim=reid_sim))
    return global_result, local_result, result

def jaccard(a_list,b_list):
    return 1.0 - float(len(set(a_list)&set(b_list)))/float(len(set(a_list)|set(b_list)))*1.0

def topk(sim, target_gallery, target_query, k=[1,5,10], dim=1, print_index=False, reid_sim = None):
    result = []
    maxk = max(k)
    size_total = len(target_query)
    if reid_sim is None:
        _, pred_index = sim.topk(maxk, dim, True, True)
        pred_labels = target_gallery[pred_index]
    else:
        K = 5
        sim = sim.cpu().numpy()
        reid_sim = reid_sim.cpu().numpy()
        pred_index = np.argsort(-sim, axis = 1)
        reid_pred_index = np.argsort(-reid_sim, axis = 1)

        q_knn = pred_index[:, 0:K]
        g_knn = reid_pred_index[:, 0:K]

        new_index = []
        jaccard_dist = np.zeros_like(sim)
        from scipy.spatial import distance
        for i, qq in enumerate(q_knn):
            for j, gg in enumerate(g_knn):
                jaccard_dist[i, j] = 1.0 - jaccard(qq, gg)
        _, pred_index = torch.Tensor(sim + jaccard_dist).topk(maxk, dim, True, True)
        pred_labels = target_gallery[pred_index]
  

    # pred
    if dim == 1:
        pred_labels = pred_labels.t()

    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))
    for topk in k:
        #correct_k = torch.sum(correct[:topk]).float()
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        result.append(correct_k * 100 / size_total)
    return result



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    """Imported from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py  L26 - L29"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    """Imported from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py  L32 - L34"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SfeNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3]):
        super(SfeNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.inplanes = 512
        self.branch2_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.branch2_layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.inplanes = 1024
        self.branch3_layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, p2, p3):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)


        # branch 1
        b13 = self.layer3(x)
        b1 = self.layer4(b13)
        b1 = self.avgpool(b1)

        # branch 2
        b2 = self.height_shuffle(x, p2)
        b2 = self.branch2_layer3(b2)
        b2 = self.branch2_layer4(b2)
        b2 = self.recover_shuffle(b2, p2)

        index_pair_list_b2 = self.get_index_pair_list(b2, p2)
        part_feature_list_b2 = [self.avgpool(b2[:, :, pair[0]:pair[1], :]).squeeze() for pair in index_pair_list_b2]

        # branch 3
        b3 = self.height_shuffle(b13, p3)
        b3 = self.branch3_layer4(b3)
        b3 = self.recover_shuffle(b3, p3)

        index_pair_list_b3 = self.get_index_pair_list(b3, p3)
        part_feature_list_b3 = [self.avgpool(b3[:, :, pair[0]:pair[1], :]).squeeze() for pair in index_pair_list_b3]
        
        # #x = x.view(x.size(0), -1)
        # #x = self.fc(x)
        #
        # return x, feature_map_v

        return b1, part_feature_list_b2, part_feature_list_b3


    def get_index_pair_list(self, x, permu):
        """
        Split feature map according to height dimension.
        :param x: Tensor with dtype torch.float32, [batchsize, num_channels, height, width]
        :param permu: List with integers, e.g. [0, 1, 2]
        
        :return: List of pairs [(start1, end1), (start2, end2)...] 
        """

        batchsize, num_channels, height, width = x.data.size()
        number_slice = len(permu)
        height_per_slice = height // number_slice
        index_pair_list = [(height_per_slice*i, height_per_slice*(i+1)) for i in range(number_slice-1)]
        index_pair_list.append((height_per_slice*(number_slice-1), height))
        return index_pair_list


    def height_shuffle(self, x, permu):
        """
        Shuffle the feature map according to height dimension.
        """

        batchsize, num_channels, height, width = x.data.size()
        result = torch.zeros(batchsize, num_channels, height, width).cuda()
        number_slice = len(permu)
        height_per_slice = height // number_slice
        index_pair_list = [(height_per_slice*i, height_per_slice*(i+1)) for i in range(number_slice-1)]
        index_pair_list.append((height_per_slice*(number_slice-1), height))
        index_pair_list_shuffled = []
        for i in range(number_slice):
            index_pair_list_shuffled.append(index_pair_list[permu[i]])
        
        start = 0
        for i in range(len(index_pair_list_shuffled)):
            index_pair = index_pair_list_shuffled[i]
            length = index_pair[1] - index_pair[0]
            result[:, :, start:(start+length), :] = x[:, :, index_pair[0]:index_pair[1], :]
            start = start + length
        return result

    def recover_shuffle(self, x, permu):
        """
        Recover the feature map to the original order.
        """
        dic = {}
        recover_permu = []
        for i in range(len(permu)):
            dic[permu[i]] = i
        all_key = list(dic.keys())
        all_key.sort()
        for i in range(len(all_key)):
            recover_permu.append(dic[all_key[i]])

        return self.height_shuffle(x, recover_permu)


class Bert(nn.Module):
    def __init__(self): 
        super(Bert, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(config.get('model_config')['language_model_path'], 'bert-base-uncased-vocab.txt'))
        modelConfig = BertConfig.from_pretrained(os.path.join(config.get('model_config')['language_model_path'], 'bert_config.json'))
        self.textExtractor = BertModel.from_pretrained(
            os.path.join(config.get('model_config')['language_model_path'], 'pytorch_model.bin'), config=modelConfig)
        
    def pre_process(self, texts):

        tokens, segments, input_masks, text_length = [], [], [], []
        for text in texts:
            text = '[CLS] ' + text + ' [SEP]'
            tokenized_text = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            if len(indexed_tokens) > 100:
                indexed_tokens = indexed_tokens[:100]
                
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            input_masks.append([1] * len(indexed_tokens))


        for j in range(len(tokens)):
            padding = [0] * (100 - len(tokens[j]))
            text_length.append(len(tokens[j])+3)
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        
        tokens = torch.tensor(tokens)
        segments = torch.tensor(segments)
        input_masks = torch.tensor(input_masks)
        text_length = torch.tensor(text_length)

        return tokens, segments, input_masks, text_length


    def forward(self, tokens, segments, input_masks):
        
        output=self.textExtractor(tokens, token_type_ids=segments,
                                 		attention_mask=input_masks)
        text_embeddings = output[0]

        return text_embeddings


class NAFS(nn.Module):
    def __init__(self):
        super(NAFS, self).__init__()

        mconfig = config.get('model_config')
        oconfig = config.get('optm_config')

        self.CMPM = oconfig['CMPM']
        self.CMPC = oconfig['CMPC']
        self.CONT = oconfig['CONT']
        self.epsilon = oconfig['epsilon']
        self.num_classes = mconfig['num_classes']
       
        self.W = Parameter(torch.randn(mconfig['feature_size'], mconfig['num_classes']))
        nn.init.xavier_uniform_(self.W.data, gain=1)

        self.part2 = mconfig['part2']
        self.part3 = mconfig['part3']

        self.image_model = SfeNet()  
        self.language_model = Bert()
        
        inp_size = 2048
    
        # shorten the tensor using 1*1 conv
        self.conv_images = nn.Conv2d(inp_size, mconfig['feature_size'], 1)
        self.conv_text = nn.Conv2d(768, mconfig['feature_size'], 1)

        # BN layer before embedding projection
        self.bottleneck_image = nn.BatchNorm1d(mconfig['feature_size'])
        self.bottleneck_image.bias.requires_grad_(False)
        self.bottleneck_image.apply(weights_init_kaiming)

        self.bottleneck_text = nn.BatchNorm1d(mconfig['feature_size'])
        self.bottleneck_text.bias.requires_grad_(False)
        self.bottleneck_text.apply(weights_init_kaiming)

        self.local_fc_text_key = nn.Linear(768, mconfig['feature_size'])
        self.local_bottleneck_text_key = nn.LayerNorm([98 + 2 + 1, mconfig['feature_size']])

        self.local_fc_text_value = nn.Linear(768, mconfig['feature_size'])
        self.local_bottleneck_text_value = nn.LayerNorm([98 + 2 + 1, mconfig['feature_size']])

        self.global_image_query = nn.Linear(mconfig['feature_size'], mconfig['feature_size'])
        self.global_image_value = nn.Linear(mconfig['feature_size'], mconfig['feature_size'])


        self.fc_p2_list = nn.ModuleList([nn.Linear(inp_size,  mconfig['feature_size']) for i in range(self.part2)])
        self.fc_p3_list = nn.ModuleList([nn.Linear(inp_size,  mconfig['feature_size']) for i in range(self.part3)])

        self.fc_p2_list_query = nn.ModuleList([nn.Linear(mconfig['feature_size'], mconfig['feature_size']) for i in range(self.part2)])
        self.fc_p2_list_value = nn.ModuleList([nn.Linear(mconfig['feature_size'], mconfig['feature_size']) for i in range(self.part2)])

        self.fc_p3_list_query = nn.ModuleList([nn.Linear(mconfig['feature_size'], mconfig['feature_size']) for i in range(self.part3)])
        self.fc_p3_list_value = nn.ModuleList([nn.Linear(mconfig['feature_size'], mconfig['feature_size']) for i in range(self.part3)])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, images, tokens, segments, input_masks, sep_tokens, sep_segments, sep_input_masks, n_sep, p2=None, p3=None, object=None, attribute=None, stage=''):

        text_features = self.language_model(sep_tokens, sep_segments, sep_input_masks)

        local_text_feat = text_features[:, 0, :]
        local_text_feat = local_text_feat.view(-1, n_sep, local_text_feat.size(1))

        b1, part_feature_list_b2, part_feature_list_b3 = self.image_model(images, p2, p3)
        text_features = self.language_model(tokens, segments, input_masks)

        global_img_feat, global_text_feat = self.build_joint_embeddings(b1, text_features[:, 0])
        global_img_feat = self.bottleneck_image(global_img_feat)
        global_text_feat = self.bottleneck_text(global_text_feat)

        local_text_feat = torch.cat((global_text_feat.unsqueeze(1), local_text_feat, text_features[:, 1:99]), dim = 1)
        local_text_key = self.local_fc_text_key(local_text_feat)
        local_text_key = self.local_bottleneck_text_key(local_text_key)

        local_text_value = self.local_fc_text_value(local_text_feat)
        local_text_value = self.local_bottleneck_text_value(local_text_value)

        for i in range(len(part_feature_list_b2)):
            part_feature_list_b2[i] = self.fc_p2_list[i](part_feature_list_b2[i])
        for i in range(len(part_feature_list_b3)):
            part_feature_list_b3[i] = self.fc_p3_list[i](part_feature_list_b3[i])

            
        global_img_query = self.global_image_query(global_img_feat)
        global_img_value = self.global_image_value(global_img_feat)

        local_img_query = torch.zeros(global_img_feat.shape[0], self.part2 + self.part3, global_img_feat.shape[1]).cuda()
        local_img_value = torch.zeros(global_img_feat.shape[0], self.part2 + self.part3, global_img_feat.shape[1]).cuda()

        for i in range(len(part_feature_list_b2)):
            local_img_query[:, i, :] = self.fc_p2_list_query[i](part_feature_list_b2[i])
        for i in range(len(part_feature_list_b3)):
            local_img_query[:, i + self.part2, :] = self.fc_p3_list_query[i](part_feature_list_b3[i])

        for i in range(len(part_feature_list_b2)):
            local_img_value[:, i, :] = self.fc_p2_list_value[i](part_feature_list_b2[i])
        for i in range(len(part_feature_list_b3)):
            local_img_value[:, i + self.part2, :] = self.fc_p3_list_value[i](part_feature_list_b3[i])

        local_img_query = torch.cat((global_img_query.unsqueeze(1), local_img_query), dim = 1)
        local_img_value = torch.cat((global_img_value.unsqueeze(1), local_img_value), dim = 1)

        return global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value


    def build_joint_embeddings(self, images_features, text_features):

        text_features = text_features.unsqueeze(2)
        text_features = text_features.unsqueeze(3)
        
        image_embeddings = self.conv_images(images_features).squeeze()
        text_embeddings = self.conv_text(text_features).squeeze()

        return image_embeddings, text_embeddings

    
    def contrastive_loss(self, i2t_similarites, t2i_similarities, labels):
        batch_size = i2t_similarites.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)
        criterion = nn.CrossEntropyLoss()

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)
       

        i2t_pred = F.softmax(i2t_similarites * config.get('model_config')['lambda_softmax'], dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(i2t_similarites * config.get('model_config')['lambda_softmax'], dim=1) - torch.log(labels_mask_norm + self.epsilon))
        sim_cos = i2t_similarites

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))

        # constrastive_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        constrastive_loss = torch.mean(torch.sum(i2t_loss, dim=1))

        return constrastive_loss, pos_avg_sim, neg_avg_sim


    def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Classfication loss(CMPC)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
        """
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W_norm = F.normalize(self.W, p=2, dim=0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

        image_logits = torch.matmul(image_proj_text, self.W_norm)
        text_logits = torch.matmul(text_proj_image, self.W_norm)

        # image_logits = torch.matmul(image_embeddings, self.W_norm)
        # text_logits = torch.matmul(text_embeddings, self.W_norm)

        '''
        ipt_loss = criterion(input=image_logits, target=labels)
        tpi_loss = criterion(input=text_logits, target=labels)
        cmpc_loss = ipt_loss + tpi_loss
        '''
        cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)

        # classification accuracy for observation
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return cmpc_loss, image_precision, text_precision

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]

        # print("batch size: " + str(batch_size))

        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        # i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        t2i_pred = F.softmax(text_proj_image, dim=1)
        # t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        sim_cos = torch.matmul(image_norm, text_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))

        return cmpm_loss, pos_avg_sim, neg_avg_sim
    
    def compute_loss(self, global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value, text_length,
                labels):
        cmpm_loss = 0.0
        cmpc_loss = 0.0
        cont_loss = 0.0
        image_precision = 0.0
        text_precision = 0.0
        neg_avg_sim = 0.0
        pos_avg_sim = 0.0
        local_pos_avg_sim = 0.0
        local_neg_avg_sim = 0.0
        if self.CMPM:
            cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(global_img_feat, global_text_feat,
                                                                         labels)
        if self.CMPC:
            cmpc_loss, image_precision, text_precision = self.compute_cmpc_loss(global_img_feat,
                                                                                global_text_feat, labels)
        if self.CONT:
            i2t_sim, t2i_sim = compute_weiTexts(local_img_query, local_img_value, local_text_key, local_text_value, text_length)
            cont_loss, local_pos_avg_sim, local_neg_avg_sim = self.contrastive_loss(i2t_sim, t2i_sim, labels)
            cont_loss = cont_loss * config.get('model_config')['lambda_cont']

        loss = cmpm_loss + cmpc_loss + cont_loss

        return cmpm_loss.item(), cmpc_loss.item(), cont_loss.item(), loss, image_precision, text_precision, pos_avg_sim, neg_avg_sim, local_pos_avg_sim, local_neg_avg_sim

