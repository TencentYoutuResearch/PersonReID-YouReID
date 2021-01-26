import torch
import torch.nn.functional as F
import numpy as np

from utils import *
from core.config import config
from train.BaseTrainer import BaseTrainer


class PartNetTrainer(BaseTrainer):


    def __init__(self):
        super(PartNetTrainer, self).__init__()

    def extract(self, test_data, model):
        model.eval()
        res = {}
        for p, val_loader in test_data.items():
            global_features, local_features, local_parts = [], [], []
            with torch.no_grad():
                paths = []
                for i, (input, _, path) in enumerate(val_loader):
                    # print(input[0])
                    input = input.cuda(non_blocking=True)
                    # compute output
                    global_feature, local_feature, local_part = model(input)
                    global_feature = F.normalize(global_feature, p=2, dim=-1)
                    local_feature = F.normalize(local_feature, p=2, dim=-1)

                    if config.get('with_flip'):
                        input_ = input.flip(3)
                        global_feature_, local_feature_, _ = model(input_)
                        global_feature_ = F.normalize(global_feature_, p=2, dim=-1)
                        local_feature_ = F.normalize(local_feature_, p=2, dim=-1)
                        global_feature = (global_feature + global_feature_) / 2
                        local_feature = (local_feature + local_feature_) / 2
                        global_feature = F.normalize(global_feature, p=2, dim=-1)
                        local_feature = F.normalize(local_feature, p=2, dim=-1)

                    global_features.append(global_feature)
                    local_features.append(local_feature)
                    local_parts.append(local_part)

                    paths.extend(path)

                global_features = torch.cat(global_features, dim=0)
                local_features = torch.cat(local_features, dim=0)
                local_parts = torch.cat(local_parts, dim=0)

                print(global_features.size(), local_features.size(), local_parts.size())
                res[p] = {
                    'global_features': global_features,
                    'local_features': local_features,
                    'local_parts': local_parts,
                    'path': paths
                }
        return res

    def eval_status(self, epoch):
        return True

    def eval_result(self, **kwargs):
        info = kwargs.get('info')
        return evaluate.eval_part_result(info, use_pcb_format=config.get('use_pcb_format'), logger=self.logger)

    def extract_and_eval(self, test_loader, model):
        res = self.extract(test_loader, model)
        mAP, rank_1 = self.eval_result(info=res)

        return mAP, rank_1


if __name__ == '__main__':
    trainer = PartNetTrainer()
    trainer.train_or_val()
