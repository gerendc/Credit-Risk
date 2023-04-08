# import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
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
    print('\nclassification report:\n', classification_report(y_true, y_pred))
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    # fig, ax = plt.subplots(figsize=(8,8))
    ax = sns.heatmap(conf_matrix, square=True, annot=True, fmt='.6g', annot_kws={'fontsize': 13}, cmap='YlGn')
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_xlabel('Prediction Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    
    return ax