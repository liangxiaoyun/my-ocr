import time
import random
import numpy as np

from mmdet.datasets.builder import PIPELINES

statistics = dict()

@PIPELINES.register_module()
class Prober(object):
    def __init__(self, alphabet_pth, key='text'):
        alphabet = ''
        with open(alphabet_pth, 'r') as f:
            alphabet = ''.join(f.readlines())
        self.alphabet = list(set(alphabet))
        self.key = key
        global statistics
        statistics.update({c: 0 for c in self.alphabet})
        self.statistics = statistics

    def __call__(self, results):
        text = results[self.key]
        for c in text:
            if c in self.statistics:
                self.statistics[c] += 1
        return results

def get_prober(x, t):
    t = max(1/x.size, t) + 1e-5
    s = (x.max() - t * x.sum()) / (t * x.size - 1)
    s = s if s > 0 else 0
    x = x + s
    return x / x.sum()

@PIPELINES.register_module()
class balanceSample(object):
    def __init__(self, max_len):
        self.max_len = int(max_len)
        global statistics
        self.statistics = statistics
        self.iter = 0

    def __call__(self, results):
        keys = []
        logits = []
        for k, v in self.statistics.items():
            keys.append(k)
            logits.append(1/(float(v)+1e-5))
        logits = get_prober(np.array(logits), 1).tolist()

        inv_stat = dict(zip(keys, logits))

        norm = sum(list(inv_stat.values()))
        l = random.randint(5, self.max_len)
        text = ''
        for _ in range(l):
            r = random.random() * norm
            temp = 0
            for key, value in inv_stat.items():
                if key not in text:
                    temp += value
                    if temp >= r:
                        text += key
                        norm -= value
                        break

        results['generate_text'] = text
        self.iter += 1
        return results


