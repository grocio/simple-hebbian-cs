#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Cosine similarity
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# It gives a random vector with the elements being 1 or -1
# e.g., [1,1,-1,1,1,-1,-1,1]
def random_posi_neg_vec(unit_num):
    zero_one_vec = np.random.randint(0,2, unit_num)
    return 1 - 2*zero_one_vec

# It stochastically changes the elements of a vector
# e.g., [1,1,1,-1,-1,-1] -> [-1,1,1,-1,1,-1]
def random_flip(vec, drift_rate):
    target_vec = np.copy(vec)
    unit_num = len(vec)
    change_num = round(unit_num*drift_rate)
    change_index = np.random.choice(unit_num, change_num)
    
    target_vec[change_index] = -1 * target_vec[change_index]

    return target_vec

# It calculates similarity between two vectors and gives a two-dementional list
def similarity_two_dim_list(two_dim_list):
    vec_num = len(two_dim_list)
    results = np.zeros((vec_num, vec_num))

    for i in range(vec_num):
        for j in range(vec_num):
            results[i][j] = cos_sim(two_dim_list[i], two_dim_list[j])

    return results

# It gives the vectors representing the drifting context
def context_vecs(position_num, drift_rate, unit_num):
    context_vecs = [0 for i in range(position_num)]
    for i in range(position_num):
        if i == 0:
            context_vecs[i] = random_posi_neg_vec(unit_num)
            # context_vecs[i] = random_flip(random_posi_neg_vec(unit_num), drift_rate)
        else:
            context_vecs[i] = random_flip(context_vecs[i-1], drift_rate)
    return context_vecs

# Simple hebbian rule for encoding
def hebbian_encoding(target_vec, cue_vec, alpha=1.0):
    return alpha*np.outer(target_vec, cue_vec)

# Simple hebbian rule for retrieval
def hebbian_retrieval(W, cue_vec):
    return np.dot(W, cue_vec)

if __name__ == '__main__':
    position_num = 5 # Ishiguro and Saito (2018) used the positions of 1 to 5
    unit_num = 2**4
    run_num = 1
    distractor_variety = 5

    single_results = np.zeros((position_num, position_num))
    multiple_results = np.zeros((position_num, position_num))

    for run in range(run_num):
        position_vecs = context_vecs(position_num, 0.3, unit_num)
        target_vecs = [random_posi_neg_vec(unit_num) for i in range(position_num)]
        distractor_vecs = [random_posi_neg_vec(unit_num) for i in range(distractor_variety)]
        distractor_vecs = [random_posi_neg_vec(unit_num) for i in range(position_num)]
        single_W = np.zeros((unit_num, unit_num))
        multiple_W = np.zeros((unit_num, unit_num))

        # Encoding
        # single distractor condition
        for i in range(position_num):
            # target
            single_W = single_W + hebbian_encoding(target_vecs[i], position_vecs[i], alpha=1.0)

            # distractor
            single_W = single_W + hebbian_encoding(distractor_vecs[0], position_vecs[i], alpha=1.0)
    
        # multiple distractor condition
        for j in range(position_num):
            # target
            multiple_W = multiple_W + hebbian_encoding(target_vecs[j], position_vecs[j], alpha=1.0)

            # distractor
            multiple_W = multiple_W + hebbian_encoding(distractor_vecs[j], position_vecs[j], alpha=1.0)

        single_output = []
        multiple_output = []

        # Retrieval
        # single distractor condition
        for i in range(position_num):
            single_output.append(hebbian_retrieval(single_W, position_vecs[i]))

        for i in range(position_num):
            for j in range(position_num):
                single_results[i][j] += cos_sim(target_vecs[i], single_output[j])
        # multiple distractor condition
        for i in range(position_num):
            multiple_output.append(hebbian_retrieval(multiple_W, position_vecs[i]))

        for i in range(position_num):
            for j in range(position_num):
                multiple_results[i][j] += cos_sim(target_vecs[i], multiple_output[j])

    single_results = single_results / run_num
    multiple_results = multiple_results / run_num

    print('single:\n', np.round(single_results, 2))
    print('multiple:\n', np.round(multiple_results, 2))
