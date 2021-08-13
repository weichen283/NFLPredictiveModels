import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_percentage_error # Requires sklearn 0.24 (December 2020), update with conda/pip if needed.
from sklearn.model_selection import GridSearchCV, train_test_split
from collections import OrderedDict
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, plot_confusion_matrix

defensive_lineman_df = pd.read_csv("Game_Logs_Defensive_Lineman.csv")
kickers_df = pd.read_csv("Game_Logs_Kickers.csv")
offence_lineman_df = pd.read_csv("Game_Logs_Offensive_Line.csv")
punters_df = pd.read_csv("Game_Logs_Punters.csv")
quarterback_df = pd.read_csv("Game_Logs_Quarterback.csv")
runningback_df = pd.read_csv("Game_Logs_Runningback.csv")
receiver_tight_end_df = pd.read_csv("Game_Logs_Wide_Receiver_and_Tight_End.csv")

datasets = [defensive_lineman_df, kickers_df, offence_lineman_df, punters_df, quarterback_df, runningback_df, receiver_tight_end_df]
processed_datasets = []
conversion = {"Outcome": {"L": int(0), "W": int(1)}, 
              "Home or Away": {"Away": int(0), "Home" : int(1)}
             }

for i in range(0, len(datasets)):
    datasets[i] = datasets[i].dropna()
    datasets[i] = datasets[i].replace(['--'], int(0))
    datasets[i] = datasets[i].drop(["Player Id", "Name", "Year", "Season", "Week", "Game Date", "Position", "Score", "Games Played", "Games Started"], axis = 1)
    datasets[i] = datasets[i].replace(conversion)
    datasets[i] = pd.get_dummies(datasets[i], columns=["Opponent"])
    for j in range(0, len(datasets[i].columns)):
        datasets[i][(datasets[i].columns)[j]] = pd.to_numeric(datasets[i][(datasets[i].columns)[j]], errors='coerce')
        datasets[i] = datasets[i].dropna()
        datasets[i][(datasets[i].columns)[j]] = datasets[i][(datasets[i].columns)[j]].astype(float)
    processed_datasets.append(datasets[i])
    
    def TreeAlgorithms(df, target_variable, test_size = 0.3, min_estimators = 50, max_estimators = 250, RANDOM_STATE = 42):
    
    # RANDOM FOREST ALGORITHM #
    
    training_set, testing_set = train_test_split(df, test_size = test_size, random_state = 42)
    ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
    ]
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(training_set.drop(target_variable, axis = 1), training_set[target_variable])

            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()
    
    optimal_number_of_estimators = input("At what point do the OOB curves stabilize?  If they don't, increase max_estimators and run again until they reach convergence: ")
    optimal_number_of_estimators = int(optimal_number_of_estimators)
    rf_object = RandomForestClassifier(n_estimators=optimal_number_of_estimators, 
                       criterion='gini', 
                       max_depth=None, 
                       min_samples_split=2, 
                       min_samples_leaf=0.001, 
                       min_weight_fraction_leaf=0.0, 
                       max_features='auto', 
                       max_leaf_nodes=None, 
                       min_impurity_decrease=0.0001, 
                       bootstrap=True, 
                       oob_score=True,  
                       n_jobs=-1, 
                       random_state=RANDOM_STATE, 
                       verbose=1, 
                       warm_start=False, 
                       class_weight='balanced'
                                    )
    rf_object.fit(training_set.drop(target_variable, axis = 1), training_set[target_variable])
    rf_pred = rf_object.predict(testing_set.drop(target_variable, axis = 1))
    rf_probs = rf_object.predict_proba(testing_set.drop(target_variable, axis = 1))[:, 1]
    roc_value = roc_auc_score(testing_set[target_variable], rf_probs)
    
    print("AUROC: ", roc_value)
    fpr, tpr, thresholds = roc_curve(testing_set[target_variable], rf_probs)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    
    importances = rf_object.feature_importances_
    indices = np.argsort(importances)[::-1] 

    f, ax = plt.subplots(figsize=(3, 8))
    plt.title("Variable Importance - Random Forest")
    sns.set_color_codes("pastel")
    sns.barplot(y=[training_set.drop(target_variable, axis = 1).columns[i] for i in indices], x=importances[indices], 
                label="Total", color="b")
    ax.set(ylabel="Variable",
           xlabel="Variable Importance (Gini)")
    sns.despine(left=True, bottom=True)
    plt.show()
    
    plot_confusion_matrix(rf_object, testing_set.drop(target_variable, axis = 1), testing_set[target_variable],
                                 cmap=plt.cm.Blues,
                                 normalize="all")
    plt.show()    
    
    # XGBOOST ALGORITHM #
    
    XGB_object = XGBClassifier(max_depth=3,                 
                            learning_rate=0.1,           
                            n_estimators=100,             
                            verbosity=1,                  
                            objective='binary:logistic',  
                            booster='gbtree',             
                            n_jobs=2,                     
                            gamma=0.001,                  
                            subsample=1,                  
                            colsample_bytree=1,           
                            colsample_bylevel=1,          
                            colsample_bynode=1,           
                            reg_alpha=1,                  
                            reg_lambda=0,                 
                            scale_pos_weight=1,           
                            base_score=0.5,               
                            random_state=RANDOM_STATE,        
                            missing=None                   
                            )
    
    number_of_trees_array = list(map(int, input("What number of trees would you like to train over?, input an integer array (Recommended: 10^2 range): ").split()))
    max_depth = list(map(int, input("What tree depths would do you want to train over?, input an integer array (Recommended: 10^0 range): ").split()))
    learning_rate = list(map(float, input("How quickly do you want your model to learn?, input a float array (Recommended: 10^-1 or 10^-2 range): ").split()))
    validation_set_proportion = input("How big do you want your validation set?, input a decimal percentage: ")
    validation_set_proportion = float(validation_set_proportion)
    param_grid = dict({'number_of_trees': number_of_trees_array,
                   'max_depth': max_depth,
                 'learning_rate' : learning_rate
                  })
    validation_set = training_set.sample(frac = validation_set_proportion, random_state = 42)
    GridXGB = GridSearchCV(XGB_object,        
                       param_grid,          
                       cv = 3,                
                       n_jobs = -1,         
                       refit = False,       
                       verbose = 1          
                      )
    
    GridXGB.fit(validation_set.drop(target_variable, axis = 1), validation_set[target_variable])
    XGB_object = XGBClassifier(max_depth=GridXGB.best_params_.get('max_depth'), 
                            learning_rate=GridXGB.best_params_.get('learning_rate'), 
                            n_estimators=GridXGB.best_params_.get('number_of_trees'),             
                            verbosity=1,                  
                            objective='binary:logistic',  
                            booster='gbtree',             
                            n_jobs=2,                     
                            gamma=0.001,                 
                            subsample=1,                  
                            colsample_bytree=1,           
                            colsample_bylevel=1,          
                            colsample_bynode=1,           
                            reg_alpha=1,                  
                            reg_lambda=0,                 
                            scale_pos_weight=1,           
                            base_score=0.5,               
                            random_state=RANDOM_STATE,        
                            missing=None                  
                            )
    
    XGB_object.fit(training_set.drop(target_variable, axis = 1), training_set[target_variable])
    xg_probs = XGB_object.predict_proba(testing_set.drop(target_variable, axis = 1))[:, 1]
    roc_value = roc_auc_score(testing_set[target_variable], xg_probs)
    
    print("AUROC: ", roc_value)
    fpr, tpr, thresholds = roc_curve(testing_set[target_variable], xg_probs)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    importances = XGB_object.feature_importances_
    indices = np.argsort(importances)[::-1] 

    f2, ax2 = plt.subplots(figsize=(3, 8))
    plt.title("Variable Importance - XGBoosting")
    sns.set_color_codes("pastel")
    sns.barplot(y=[training_set.drop(target_variable, axis = 1).columns[i] for i in indices], x=importances[indices], 
                label="Total", color="b")
    ax2.set(ylabel="Variable",
           xlabel="Variable Importance (Gini)")
    sns.despine(left=True, bottom=True)
    plt.show()
    
    plot_confusion_matrix(XGB_object, testing_set.drop(target_variable, axis = 1), testing_set[target_variable],
                                 cmap=plt.cm.Blues,
                                 normalize="all")
    plt.show() 
    
    
    ### EXAMPLE USE ###
    # TreeAlgorithms(processed_datasets[0], "Outcome", test_size = 0.3, min_estimators = 50, max_estimators = 250)
    # At what point do the OOB curves stabilize?  If they don't, increase max_estimators and run again until they reach convergence: 200
    # What number of trees would you like to train over?, input an integer array (Recommended: 10^2 range): 350 400 450 500
    # What tree depths would do you want to train over?, input an integer array (Recommended: 10^0 range): 3 4 5 6 7
    # How quickly do you want your model to learn?, input a float array (Recommended: 10^-1 or 10^-2 range): 0.01 0.1 0.2
    # How big do you want your validation set?, input a decimal percentage: 0.3
