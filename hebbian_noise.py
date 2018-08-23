#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rnd

# Cosine similarity
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# It gives a random vector with the elements being 1 or -1
# e.g., [1,1,-1,1,1,-1,-1,1]
def random_posi_neg_vec(unit_num):
    zero_one_vec = np.random.randint(0,2, unit_num)
    return 1 - 2*zero_one_vec

# Simple hebbian rule for encoding
def hebbian_encoding(target_vec, cue_vec, alpha=1.0):
    return alpha*np.outer(target_vec, cue_vec)

# Simple hebbian rule for retrieval
def hebbian_retrieval(W, cue_vec):
    return np.dot(W, cue_vec)

if __name__ == '__main__':
    unit_num = 2**4
    run_num = 50

    noise_repetition_results = 0
    noise_change_results = 0

    for run in range(run_num):
        target_vec = random_posi_neg_vec(unit_num)
        cue_vec = random_posi_neg_vec(unit_num)

        # noise repetition
        W_repetition = np.zeros((unit_num, unit_num))
        W_repetition = W_repetition + hebbian_encoding(target_vec, cue_vec, alpha=1.0) +\
                                    2*rnd.randn(unit_num, unit_num)

        # noise change
        W_change = np.zeros((unit_num, unit_num))
        W_change = W_change + hebbian_encoding(target_vec, cue_vec, alpha=1.0) +\
                        rnd.randn(unit_num, unit_num) + rnd.randn(unit_num, unit_num)

        # Retrieval
        noise_repetition_results += cos_sim(target_vec, hebbian_retrieval(W_repetition, cue_vec))
        noise_change_results += cos_sim(target_vec, hebbian_retrieval(W_change, cue_vec))

    noise_repetition_results /= run_num
    noise_change_results /= run_num

    print('Accuracy for the Repetition condition: ', noise_repetition_results)
    print('Accuracy for the Change condition:     ', noise_change_results)

    """
    Accuracy for the Repetition condition:  0.894983570423
    Accuracy for the Change condition:      0.946074806177
    """
