# import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced

# seed
np.random.seed(42)

# set figure size
plt.rcParams["figure.figsize"] = (15,8)

# set seaborn style
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, palette='tab10')

# ------------------------------------------------------------------------------------------------------------- #
# Histogram Plot
def plotHistogram(data, x, xlabel, ylabel, title):
    plt.figure()
    ax = sns.histplot(data=data, x=x, kde=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    return ax

# ------------------------------------------------------------------------------------------------------------- #
# Cross validation Models
def checkModel(models, X, y, kfold, metric='f1'):
    model_name = []
    result = []
    
    for name, model in models:
        cv_score = cross_val_score(model, X, y, cv=kfold, scoring=metric)
        model_name.append(name)
        result.append(cv_score)
        print(f'{name} \t | cv_score_mean: {cv_score.mean()} \t | cv_score_std: {cv_score.std()}')

    return model_name, result

# ------------------------------------------------------------------------------------------------------------- #
# Boxplot for Model Comparison
def boxplotModel(model_name, result, metric_name='f1'):
    results_df = pd.DataFrame(np.asarray(result).T, columns=model_name)

    ax = sns.boxplot(data=results_df, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    ax.set_xlabel('Models')
    ax.set_ylabel(f'Metric {metric_name} Score')
    ax.set_title('Models Comparison', fontsize=14)
    
    return ax

# ------------------------------------------------------------------------------------------------------------- #
# Metrics for model evaluation
def evaluateModel(y_true, y_pred, model):
    print(f'{model} Evaluation')
    print('-'*70)
    print('Accuracy Score:', accuracy_score(y_true, y_pred))
    print('Balanced Accuracy Score:', balanced_accuracy_score(y_true, y_pred))
    print('Roc Auc Score:', roc_auc_score(y_true, y_pred))
    print('Precision Score:', precision_score(y_true, y_pred))
    print('Recall Score:', recall_score(y_true, y_pred))
    print('F1 Score:', f1_score(y_true, y_pred))
    print('\nclassification report:\n', classification_report(y_true, y_pred))
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    # fig, ax = plt.subplots(figsize=(8,8))
    ax = sns.heatmap(conf_matrix, square=True, annot=True, fmt='.6g', annot_kws={'fontsize': 13}, cmap='YlGn')
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_xlabel('Prediction Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    
    return ax

# ------------------------------------------------------------------------------------------------------------- #
def plotFeatureImportance(importance, names, model_type):
    feature_importance_array = np.array(importance)
    feature_names_array = np.array(names)

    data={'feature_names':feature_names_array,'feature_importance':feature_importance_array}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    # plt.figure(figsize=(10,8))
    ax = sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    ax.set_title(model_type + ' Feature Importance Score', fontsize=14)
    ax.set_xlabel('Feature Importance Score')
    ax.set_ylabel('Feature Name')
    
    return ax

# ------------------------------------------------------------------------------------------------------------- #

def dt_grid_search(X, y, nfolds, scoring):
    #create a dictionary of all values we want to test
    param_grid ={'criterion':['gini','entropy'],
                 'max_depth': np.arange(10, 50),
                 'max_features': [None, 'sqrt'],
                 }
    # decision tree model
    dt_model = DecisionTreeClassifier(random_state=42)
    
    # using gridsearch to test all values
    dt_gscv = GridSearchCV(dt_model, param_grid, cv=nfolds, scoring=scoring)
    
    # fit model to data
    dt_gscv.fit(X, y)
    
    return dt_gscv.best_params_

# ------------------------------------------------------------------------------------------------------------- #

def rf_grid_search(X, y, nfolds, scoring):
    #create a dictionary of all values we want to test
    param_grid = { 
    'n_estimators': [100, 150, 175, 200],
    'max_features': ['sqrt', None],
    'max_depth' : [10, 15, 20, 25],
    'criterion' :['gini', 'entropy']
    }
    
    # random forest model
    rf_model = RandomForestClassifier(random_state=42)
    
    #use gridsearch to test all values
    rf_gscv = GridSearchCV(rf_model, param_grid, cv=nfolds, scoring=scoring)
    
    #fit model to data
    rf_gscv.fit(X, y)
    
    return rf_gscv.best_params_