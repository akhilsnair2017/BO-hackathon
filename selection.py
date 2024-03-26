import pandas as pd
import numpy as np
from scipy.stats import norm
from acquisition import AcquisitionFunction
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

class CandidateSelection:
    common_error_msg = 'Acquisition function "{acquisition_function}" not supported'

    def __init__(self, X_train, y_train, X_test, acquisition_function):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.acquisition_function = acquisition_function

    def null_model_select(self):
        if self.acquisition_function.lower() == 'random':
            next_idx = np.random.choice(self.X_test.index, size=1)[0]
        elif self.acquisition_function.lower() == "distance":
            scaler = StandardScaler().fit(self.X_train)
            X_train = scaler.fit_transform(self.X_train)
            X_test = scaler.transform(self.X_test)
            D = euclidean_distances(X_train, X_test)
            next_idx = np.argmax(np.min(D, axis=1))
        else:
            raise NotImplementedError(self.common_error_msg.format(acquisition_function=self.acquisition_function))

        return next_idx

    def model_driven_select(self, pred, unct, current_best, iteration_no, epsilon=None, target_window=None):
        acq = AcquisitionFunction(mean_pred=pred, unct=unct, current_best=current_best)
        
        if self.acquisition_function.lower() == 'pe':
            next_idx = np.argmin(pred)
        elif self.acquisition_function.lower() == "mu":
            next_idx = np.argmax(unct)
        elif self.acquisition_function.lower() == "ei":
            scores = acq.EI(epsilon)
            next_idx = np.argmax(scores)
        elif self.acquisition_function.lower() == "mli":
            scores = acq.MLI(target_window)
            next_idx = np.argmax(scores)
        elif self.acquisition_function.lower() == "lcb":
            scores = acq.LCB(epsilon)
            next_idx = np.argmax(scores)
        else:
            raise NotImplementedError(self.common_error_msg.format(acquisition_function=self.acquisition_function))
        
        # Write scores to file named by the acquisition function
        #df_scores = pd.DataFrame(scores, columns=['Scores'])
    
    # Save DataFrame to a CSV file named after the iteration number
        #df_scores.to_csv(f'{self.acquisition_function}_{iteration_no}_scores.csv', index=False)
 
        return next_idx
