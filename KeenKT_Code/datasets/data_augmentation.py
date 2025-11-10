import torch
import random

def augment_sequence(q, c, r, max_shift=3):
    """
    对答题序列进行随机扰动
    :param q: 问题序列
    :param c: 概念序列
    :param r: 回答结果序列
    :param max_shift: 最大随机位移量
    :return: 增强后的序列
    """
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
        q = torch.cat([torch.zeros(shift, dtype=q.dtype), q[:-shift]])
        c = torch.cat([torch.zeros(shift, dtype=c.dtype), c[:-shift]])
        r = torch.cat([torch.zeros(shift, dtype=r.dtype), r[:-shift]])
    elif shift < 0:
        q = torch.cat([q[-shift:], torch.zeros(-shift, dtype=q.dtype)])
        c = torch.cat([c[-shift:], torch.zeros(-shift, dtype=c.dtype)])
        r = torch.cat([r[-shift:], torch.zeros(-shift, dtype=r.dtype)])
    return q, c, r