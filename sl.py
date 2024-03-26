import json
import gzip
import pandas as pd
import numpy as np
from typing import List,Dict,Union
from estimators import Estimator,get_pred_unct
from get_phases import dft_run
from selection import CandidateSelection
from acquisition import AcquisitionFunction
from get_phases import dft_run

def sequential_learning(train_file, test_file, non_feature_columns, prop_label, model, acquisition_function, iteration_no, model_kwargs):
    non_feature_columns.append(prop_label)
    train_df, test_df = pd.read_csv(train_file), pd.read_csv(test_file)
    feature_cols = [col for col in train_df.columns if col not in non_feature_columns]
    X_train, y_train = train_df[feature_cols], train_df[prop_label].values
    X_test = test_df[feature_cols]
    base_X_test = test_df.loc[X_test.index].reset_index(drop=True)
    current_best = y_train.min()
    null_model_aqfns = ['random', 'distance']
    print(f"Acquisition Function : {acquisition_function}")
    print(f"SL iteration: {iteration_no}")
    print("Length of Design space being predicted:", len(X_test))
    print("Current best value:", current_best)

    if model == 'sisso':
        model_kwargs['iter_no'] = iteration_no
        model_kwargs['test_data'] = test_df

    select_next_candidate = CandidateSelection(X_train, y_train, X_test, acquisition_function)

    if acquisition_function in null_model_aqfns:
        next_idx = select_next_candidate.null_model_select()
    else:
        # Compute pred and unct separately
        pred, unct = get_pred_unct(model, X_train, y_train, X_test, model_kwargs)
        
        # Adjust the parameters based on your AcquisitionFunction class
        
        next_idx = select_next_candidate.model_driven_select(pred, unct, current_best,iteration_no, epsilon=0.01)

    print("Next candidate index from design space:", next_idx)
    next_candidate = X_test.loc[next_idx]

    print("Next candidate material from design space:", test_df.loc[next_idx].oxide)

    dft_estimator= dft_run(test_df.iloc[next_idx].formula,'done.db','phases.db','undone.db',iteration_no)
    dft_estimator.write_phases()
#        new_y_value=dft_estimator.dft_run()
#
#         query_result[base_X_test.loc[next_idx].oxide]=new_y_value

        #X_train = X_train.append(next_candidate, ignore_index=True)
#        if isinstance(next_candidate, pd.Series):
#            next_candidate = pd.DataFrame(next_candidate).T

        # Ensure column names are strings before concatenation
#        next_candidate.columns = next_candidate.columns.astype(str)

#        X_train = pd.concat([X_train, next_candidate], ignore_index=True)
#        y_train = np.append(y_train, new_y_value)

#        if target_window[0]<=new_y_value<=target_window[1]:
#            no_discovered_mats+=1

#        current_best = np.min(y_train)
#        print("Updated best value:", current_best)
#        min_values.append(current_best)
#        discovered_mats_per_iteration.append(no_discovered_mats)

#        X_test = X_test.drop(next_idx).reset_index(drop=True)
#        base_X_test = base_X_test.drop(next_idx).reset_index(drop=True)
#        y_test = np.delete(y_test, next_idx)

#        print("Length of updated design space ", len(X_test), "\n")


iteration_no=26
train_df=f'train_{iteration_no}.csv'
test_df=f'test_{iteration_no}.csv'
non_feature_columns=['oxide','formula']
model='sisso'
acquisition_function='mu'
prop_label='g_pbx (eV/atom)'
model_kwargs={'workdir':'sisso_run','prop_label':'g_pbx','prop_unit':"eV/atom","n_dim":2,"n_res":5,"n_sis":50,"max_rung":2,"cv_fold":10,'n_bootstrap':10}

sequential_learning(train_df,test_df,non_feature_columns,prop_label,model,acquisition_function,iteration_no,model_kwargs)

#
#def get_statistics(df,non_feature_columns,prop_label,train_size,split_method,model,acquisition_function,num_iterations,num_repeats,target_window=None,model_kwargs=None,**kwargs):
#    #all_min_values = []
#    num_discovered_mats=[]
#    for i in range(num_repeats):
#        if model=='sisso':
#            model_kwargs['workdir']=f'sisso_run/{acquisition_function}/run_{i}'
#        print(f"Run: {i + 1}")
#        # Create a copy of the original data for each repetition
#        discover_mats = run_sequential_learning(df,non_feature_columns,prop_label,train_size,split_method,model,acquisition_function,num_iterations,target_window,model_kwargs,**kwargs)
##        print(f"Run: {i+1},min_values: {min_values}")
#        print(f"Run: {i+1},dis_mats: {discover_mats}")
#        #all_min_values.append(min_values)
#        num_discovered_mats.append(discover_mats)
#    #print(all_min_values)
#    print(num_discovered_mats)
#    # Calculate mean and standard deviation across runs for each iteration
##     print(num_discovered_mats)
#    #mean_min_values = np.mean(all_min_values, axis=0)
#    #std_min_values = np.std(all_min_values, axis=0)
#    mean_dis_mats = np.mean(num_discovered_mats,axis=0)
#    std_dis_mats = np.std(num_discovered_mats,axis=0)   
#    return mean_dis_mats, std_dis_mats
#
#def plot_results(df,non_feature_columns,prop_label,train_size,split_method,model,num_iterations,num_repeats,target_window=None,**kwargs):
#    target_mats=df[(df[prop_label]>=target_window[0]) & (df[prop_label]<=target_window[1])]
#    no_targets=len(target_mats)
#    aq_fns=['random','mu','mli']
#    labels=['Random','MU','MLI']
#    colors = ["gray","green","brown"]
#    if model=='sisso':
#        model_kwargs={'prop_label':'g_pbx','prop_unit':"eV/atom","n_dim":2,"n_res":5,
#                   "n_sis":50,"max_rung":1,"cv_fold":10,'n_bootstrap':10}
#    elif model=='rf':
#        model_kwargs={'base_estimator':'sklearn'}
#    elif model=='gpr':
#        model_kwargs={}
#    fig,ax=plt.subplots(figsize=(8,6))
#    ax.axhline(y=no_targets, color='black', linestyle='--',linewidth=1.5)
#    for i,aq in enumerate(aq_fns):
#        mean_mats,std_mats=get_statistics(df,non_feature_columns,prop_label,train_size,split_method,model,f'{aq}',num_iterations,num_repeats,target_window,model_kwargs,**kwargs)
#        print(f"Mean Mats: {mean_mats}\n Mean Stds: {std_mats}")
##        ax.plot(range(1, len(mean_values)+1), mean_values, linestyle='-', label=f'{aq}',color=colors[i])
##        ax.fill_between(range(1, len(mean_values) +1 ), mean_values - std_values, mean_values + std_values, alpha=0.2,color=colors[i])
#        ax.plot(range(1, len(mean_mats)+1), mean_mats, linestyle='-', label=f'{labels[i]}',color=colors[i])
#        ax.fill_between(range(1, len(mean_mats)+1), mean_mats - std_mats, mean_mats + std_mats, alpha=0.4,color=colors[i])
#
##    ax.set_xlabel('Number of Iterations',fontsize=20)
##    ax.set_ylabel(r"Minimum $\Delta H_{f}$ (eV/atom)",fontsize=20)
##    #ax.set_title('Property minimization')
#    ax.legend(fontsize=22)
##    ax.set_xlim([1,100])
#    ax.tick_params(axis='x',labelsize=20)
#    ax.tick_params(axis='y',labelsize=20)
#    ax.set_xlabel('Number of Iterations',fontsize=24)
#    ax.set_ylabel('No. of Discovered Materials',fontsize=24)
#    ax.set_title('Material Discovery',fontsize=24)
#    ax.set_xlim([0,200])
#    ax.set_xticks([0,50,100,150,200])
#    ax.set_ylim([0,155])
#    ax.set_yticks([0,50,100,150])
#    for spine in ax.spines.values():
#        spine.set_linewidth(2)
#    fig.tight_layout()
#    return fig
#
#figure=plot_results(df,['oxide','formula'],'g_pbx (eV/atom)',10,'random_split','sisso',200,10,[0,0.5],position='in',std=1)
#figure.savefig('SISSO_gpbx_200_random_10.png',dpi=300)
