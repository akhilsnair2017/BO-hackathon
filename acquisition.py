#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
from scipy.stats import norm
from typing import List
import numpy as np

class AcquisitionFunction:

    """class of analytical acqusition functions"""

    def __init__(self, mean_pred: np.ndarray, unct: np.ndarray,current_best:float):
        self.mean_pred = mean_pred
        self.unct = unct
        self.current_best = current_best

    #Probability Improvement
    def PI(self,epsilon) -> np.ndarray:
        epsilon = 0
        zzval = (self.current_best-self.mean_pred-epsilon) / self.unct
        return norm.cdf(zzval)

    #Expected Improvement    
    def EI(self, epsilon: float) -> np.ndarray:
        assert self.mean_pred.shape == self.unct.shape
        #zzval = (self.mean_pred - self.current_best-epsilon)/self.unct
        #EI = (self.mean_pred - self.current_best - epsilon) * norm.cdf(zzval) + self.unct * norm.pdf(zzval)
        zzval = (self.current_best - self.mean_pred-epsilon) / self.unct
        EI = (self.current_best - self.mean_pred - epsilon) * norm.cdf(zzval) + self.unct * norm.pdf(zzval)
        EI[self.unct <= 0] = 0.0
        return EI
    #lower confidence bound
    def LCB(self, epsilon:float) -> np.ndarray:
        return -self.mean_pred + epsilon * self.unct

    #upper confidence bound
    def UCB(self, epsilon: float) -> np.ndarray:
        return self.mean_pred + epsilon * self.unct

    #maximum likelyhood of improvement
    def MLI(self, target_window: List[float]) -> np.ndarray:
        x1, x2 = target_window
        norm_dist = norm(loc=self.mean_pred, scale=self.unct)
        return norm_dist.cdf(x2) - norm_dist.cdf(x1)


