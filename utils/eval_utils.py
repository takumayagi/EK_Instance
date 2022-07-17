#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import os.path as osp
import pickle

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix


def calc_unsupervised_accuracy(gt_labels, pr_labels):
  D = max(pr_labels.max(), gt_labels.max()) + 1
  W = np.zeros((D, D), dtype=np.int64)
  # Confusion matrix.
  for i in range(pr_labels.size):
    W[pr_labels[i], gt_labels[i]] += 1

  ind = linear_assignment(-W)
  acc = sum([W[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / pr_labels.size

  return acc


def match_labels(gt_labels, pr_labels):
  D = max(pr_labels.max(), gt_labels.max()) + 1
  W = np.zeros((D, D), dtype=np.int64)
  # Confusion matrix.
  for i in range(pr_labels.size):
    W[pr_labels[i], gt_labels[i]] += 1

  ind = linear_assignment(-W)
  return ind


def calc_results(gt_labels, pr_labels):
  # Evaluation
  nmi = metrics.normalized_mutual_info_score(gt_labels, pr_labels)
  ami = metrics.adjusted_mutual_info_score(gt_labels, pr_labels)
  ari = metrics.adjusted_rand_score(gt_labels, pr_labels)
  acc = calc_unsupervised_accuracy(gt_labels, pr_labels)
  hom = metrics.homogeneity_score(gt_labels, pr_labels)
  com = metrics.completeness_score(gt_labels, pr_labels)
  v_measure = metrics.v_measure_score(gt_labels, pr_labels)

  f_paired = pairwise(gt_labels, pr_labels)
  f_bcubed = bcubed(gt_labels, pr_labels)

  return {
    "nmi": nmi,
    "ami": ami,
    "ari": ari,
    "acc": acc,
    "hom": hom,
    "com": com,
    "v_measure": v_measure,
    "fp": f_paired,
    "fb": f_bcubed
  }


def report_results(gt_labels, pr_labels, logger):
  # Evaluation
  nmi = metrics.normalized_mutual_info_score(gt_labels, pr_labels)
  ami = metrics.adjusted_mutual_info_score(gt_labels, pr_labels)
  ari = metrics.adjusted_rand_score(gt_labels, pr_labels)
  acc = calc_unsupervised_accuracy(gt_labels, pr_labels)
  hom = metrics.homogeneity_score(gt_labels, pr_labels)
  com = metrics.completeness_score(gt_labels, pr_labels)
  v_measure = metrics.v_measure_score(gt_labels, pr_labels)

  f_paired = pairwise(gt_labels, pr_labels)
  f_bcubed = bcubed(gt_labels, pr_labels)

  # logger.info(f"NMI: {nmi:.3f}")
  logger.info(f"AMI: {ami:.3f}")
  # logger.info(f"ARI: {ari:.3f}")
  logger.info(f"ACC: {acc:.3f}")
  logger.info(f"Homogeneity: {hom:.3f}")
  logger.info(f"Completeness: {com:.3f}")
  logger.info(f"V-measure: {v_measure:.3f}")
  logger.info(f"Fp: {f_paired:.3f}")
  logger.info(f"Fb: {f_bcubed:.3f}")

  return {
    "nmi": nmi,
    "ami": ami,
    "ari": ari,
    "acc": acc,
    "hom": hom,
    "com": com,
    "v_measure": v_measure,
    "fp": f_paired,
    "fb": f_bcubed
  }


def _check(gt_labels, pred_labels):
    if gt_labels.ndim != 1:
        raise ValueError("gt_labels must be 1D: shape is %r" %
                         (gt_labels.shape, ))
    if pred_labels.ndim != 1:
        raise ValueError("pred_labels must be 1D: shape is %r" %
                         (pred_labels.shape, ))
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            "gt_labels and pred_labels must have same size, got %d and %d" %
            (gt_labels.shape[0], pred_labels.shape[0]))
    return gt_labels, pred_labels


def _get_lb2idxs(labels):
    lb2idxs = {}
    for idx, lb in enumerate(labels):
        if lb not in lb2idxs:
            lb2idxs[lb] = []
        lb2idxs[lb].append(idx)
    return lb2idxs


def _compute_fscore(pre, rec):
    return 2. * pre * rec / (pre + rec)


def fowlkes_mallows_score(gt_labels, pred_labels, sparse=True):
    ''' The original function is from `sklearn.metrics.fowlkes_mallows_score`.
        We output the pairwise precision, pairwise recall and F-measure,
        instead of calculating the geometry mean of precision and recall.
    '''
    n_samples, = gt_labels.shape

    c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel()**2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel()**2) - n_samples

    avg_pre = tk / pk
    avg_rec = tk / qk
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore


def pairwise(gt_labels, pred_labels, sparse=True):
    _check(gt_labels, pred_labels)
    return fowlkes_mallows_score(gt_labels, pred_labels, sparse)[2]


def bcubed(gt_labels, pred_labels):
    _check(gt_labels, pred_labels)

    gt_lb2idxs = _get_lb2idxs(gt_labels)
    pred_lb2idxs = _get_lb2idxs(pred_labels)

    num_lbs = len(gt_lb2idxs)
    pre = np.zeros(num_lbs)
    rec = np.zeros(num_lbs)
    gt_num = np.zeros(num_lbs)

    for i, gt_idxs in enumerate(gt_lb2idxs.values()):
        all_pred_lbs = np.unique(pred_labels[gt_idxs])
        gt_num[i] = len(gt_idxs)
        for pred_lb in all_pred_lbs:
            pred_idxs = pred_lb2idxs[pred_lb]
            n = 1. * np.intersect1d(gt_idxs, pred_idxs).size
            pre[i] += n**2 / len(pred_idxs)
            rec[i] += n**2 / gt_num[i]

    gt_num = gt_num.sum()
    avg_pre = pre.sum() / gt_num
    avg_rec = rec.sum() / gt_num
    fscore = _compute_fscore(avg_pre, avg_rec)

    return fscore
