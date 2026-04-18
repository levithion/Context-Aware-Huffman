import main
import math
import argparse
import sys
import time

def calc_shannon_entropy(freq_map):
    total = sum(freq_map.values())
    if total == 0: return 0.0
    return -sum((f/total) * math.log2(f/total) for f in freq_map.values() if f > 0)

def calc_conditional_entropy(context_freqs):
    total_tokens = sum(sum(f.values()) for f in context_freqs.values())
    if total_tokens == 0: return 0.0
    cond_entropy = 0.0
    for ctx, freq_map in context_freqs.items():
        ctx_total = sum(freq_map.values())
        ctx_prob = ctx_total / total_tokens
        ctx_entropy = calc_shannon_entropy(freq_map)
        cond_entropy += ctx_prob * ctx_entropy
    return cond_entropy
