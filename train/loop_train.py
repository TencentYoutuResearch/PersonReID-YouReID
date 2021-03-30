import os
import yaml

with open('../config.yaml') as rf:
    cfg = yaml.load(rf)

for i in [1, 2, 4]:
    for j in range(5):
        path = cfg['yaml']
        with open(path) as sf:
            param_cfg = yaml.load(sf)
            param_cfg['dataset_config']['sample_image_per_class'] = i
        with open(path, 'w') as swf:
            yaml.dump(param_cfg, swf)
        cfg['task_id'] = '../../snapshot/yewu/softmax_id%d_%d' % (i, j)
        with open('../config.yaml', 'w') as wf:
            yaml.dump(cfg, wf)
    os.system('./cmd.sh')