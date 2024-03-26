from sissopp.sklearn import SISSORegressor, mean_squared_error, make_scorer, regression_metric
from sissopp.py_interface.import_dataframe import strip_units
from sissopp.postprocess.load_models import load_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from lolopy.learners import RandomForestRegressor as LoloRandomForestRegressor
import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import norm
#Defining estimators for sequential learning 

class Estimator:
    """Estimators for prediction"""

    def __init__(self, X_train, y_train, X_test,  model_kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model_kwargs = model_kwargs

    def GPR(self):
        scaler = StandardScaler().fit(self.X_train)
        X_train_scaled = scaler.transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
#        gpr = GaussianProcessRegressor(kernel=Matern(), alpha=1e-4, n_restarts_optimizer=20, random_state=13)
        gpr=GaussianProcessRegressor(**self.model_kwargs)
        gpr.fit(X_train_scaled, self.y_train)
        y_pred, std = gpr.predict(X_test_scaled, return_std=True)
        return y_pred, std

    def RF(self):
        base_estimator = self.model_kwargs.get('base_estimator')
        if base_estimator == 'sklearn':
            rf = RandomForestRegressor(max_depth=10, min_samples_split=4, n_estimators=len(self.X_train), random_state=13)
            rf.fit(self.X_train, self.y_train)
            preds = [tree.predict(self.X_test.values) for tree in rf.estimators_]
            mean_pred, sigma = np.mean(preds, axis=0), np.std(preds, axis=0)
            return mean_pred, sigma

        elif base_estimator == 'lolopy':
            rf = LoloRandomForestRegressor(max_depth=10, num_trees=len(self.X_train), min_leaf_instances=4)
            rf.fit(np.array(self.X_train), self.y_train)
            mean_pred, sigma = rf.predict(self.X_test, return_std=True)
            return mean_pred, sigma

    def sisso(self):
        sisso_est = SISSORegressor(
            prop_label=self.model_kwargs['prop_label'],
            prop_unit=self.model_kwargs['prop_unit'],
            allowed_ops=["add", "sub", "abs_diff", "mult", "div", "inv", "exp", "sq",
                          "cb", "sqrt", "cbrt", "abs", "six_pow", "neg_exp"],
            n_dim=self.model_kwargs['n_dim'],
            max_rung=self.model_kwargs['max_rung'],
            n_sis_select=self.model_kwargs['n_sis'],
            n_residual=self.model_kwargs['n_res'])
#         X_train=pd.DataFrame(self.X_train,columns=self.feature_columns)
        n_samples = self.X_train.shape[0]
        test_df=self.model_kwargs['test_data']
        bs_preds = {}
        for i in range(self.model_kwargs['n_bootstrap']):  # Changed 'n_bootstrap' to 'self.n_bootstrap'
            print(f"bootstrap: {i}")
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = self.X_train.iloc[bootstrap_indices, :]  # Fixed the missing 'self.' in front of 'X_train'
            y_bootstrap = self.y_train[bootstrap_indices]  # Fixed the missing 'self.' in front of 'y_train'
            sisso_est.workdir = f"{self.model_kwargs['workdir']}/iter_{self.model_kwargs['iter_no']}/bs_{i}"
            sisso_est.fit(X_bootstrap, y_bootstrap)
            pred = sisso_est.predict(self.X_test)

            model=load_model(f"{sisso_est.workdir}/models/train_dim_{self.model_kwargs['n_dim']}_model_0.dat")
            eval_pred = model.eval_many(self.X_test)
            sample_ids = test_df['oxide'].to_list()
            data_predict = strip_units(test_df.set_index('oxide'))
            model.prediction_to_file(f"{sisso_est.workdir}/pred_dim_{self.model_kwargs['n_dim']}_model_0.dat", np.full(len(sample_ids), np.nan), data_predict.iloc[:, 1:].loc[sample_ids,:], sample_ids,[])
            bs_preds[f"bs_{i}"] = pred
        array_data = np.array(list(bs_preds.values()))
        mean_pred = np.mean(array_data, axis=0)
        sigma = np.std(array_data, axis=0)
        index = np.array(test_df['oxide'].astype(str))
        result_data = np.column_stack((index, mean_pred, sigma))
        result_file_path = f"{self.model_kwargs['workdir']}/iter_{self.model_kwargs['iter_no']}/pred_unct.csv"
        header_line = 'oxide,pred,sigma'
        np.savetxt(result_file_path, result_data, header=header_line, delimiter=',', comments='', fmt='%s,%.8f,%.8f')
        return mean_pred, sigma

def get_pred_unct(model, X_train, y_train, X_test, model_kwargs=None):
    est=Estimator(X_train, y_train, X_test, model_kwargs)
    if model.lower() == 'rf':
        pred, unct = est.RF()
    elif model.lower() == 'gpr':
        pred, unct = est.GPR()
    elif model.lower() == 'sisso':
        pred, unct = est.sisso()
    return pred, unct

if __name__=='__main__':
    pass
