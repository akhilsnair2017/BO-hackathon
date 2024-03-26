import numpy as np
import os
import pandas as pd
from scipy.stats import norm
def EI(mean_pred, unct, current_best, epsilon: float) -> np.ndarray:
    assert mean_pred.shape == unct.shape
    zzval = (mean_pred - current_best-epsilon)/unct
    EI = (mean_pred - current_best - epsilon) * norm.cdf(zzval) + unct * norm.pdf(zzval)
    #zzval = (current_best-mean_pred-epsilon)/unct
    #EI = (current_best - mean_pred-epsilon) * norm.cdf(zzval) + unct * norm.pdf(zzval)
    EI[unct <= 0] = 0.0
    return EI
current_best=-2.925016536045288
pred_unct = pd.read_csv('sisso_run/iter_2/pred_unct.csv')
mean_pred,unct=pred_unct['pred'],pred_unct['sigma']
scores=EI(mean_pred,unct,current_best,0.01)
print(scores)
