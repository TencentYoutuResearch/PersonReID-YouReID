#conding=utf-8


def select_network(model_name):
    """"""
    if model_name == 'baseline':
        from nets.pcb.baseline import Baseline
        return Baseline
    elif model_name == 'cacenet':
        from nets.cacenet.cacenet import CACENET
        return CACENET
    elif model_name == 'mgn':
        from nets.mgn import MGN
        return MGN
    else:
        raise ValueError('暂不支持')