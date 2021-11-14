#!/usr/bin/env python
# coding: utf-8

# # Task 3
# ---
# Binary classification
# 
# [Kaggle](https://www.kaggle.com/dipayanbiswas/parkinsons-disease-speech-signal-features)

# # Dataset description

# __Context__
# 
# This dataset is collected from UCI Machine Learning Repository through the following link: [click](https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification#)
# 
# __Data Set Information:__
# 
# The data used in this study were gathered from 188 patients with PD (107 men and 81 women) with ages ranging from 33 to 87 (65.1±10.9) at the Department of Neurology in Cerrahpaya Faculty of Medicine, Istanbul University. The control group consists of 64 healthy individuals (23 men and 41 women) with ages varying between 41 and 82 (61.1±8.9). During the data collection process, the microphone is set to 44.1 KHz and following the physician's examination, the sustained phonation of the vowel /a/ was collected from each subject with three repetitions.
# 
# __Attribute Information:__
# 
# Various speech signal processing algorithms including Time Frequency Features, Mel Frequency Cepstral Coefficients (MFCCs), Wavelet Transform based Features, Vocal Fold Features and TWQT features have been applied to the speech recordings of Parkinson's Disease (PD) patients to extract clinically useful information for PD assessment. [Related paper](https://www.sciencedirect.com/science/article/abs/pii/S1568494618305799?via%3Dihub)
# 
# Attribute description:
# 
# - Baseline Features: Col3 to Col23
# - Intensity Parameters: Col24 to Col26
# - Formant Frequencies: Col27 to Col30
# - Bandwidth Parameters: Col31 to Col34
# - Vocal Fold: Col35 to Col56
# - MFCC: Col57 to Col140
# - Wavelet Features: Col141 to Col322
# - TQWT Features: Col323 to Col754
# - Class: Col755
# 
# [Similar paper](https://arxiv.org/ftp/arxiv/papers/1905/1905.00377.pdf)

# # Import required libraries

# In[1]:


import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn import set_config
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.utils import resample

from sklearn_pandas import DataFrameMapper, gen_features

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

from IPython.display import HTML, display

from typing import Tuple

import plotly.express as px

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme()


# In[2]:


PALETTE = sns.color_palette("Set2")
RANDOM_STATE = 42


# In[3]:


sns.set_context("paper", rc={"font.size":12, 
                             "figure.titlesize":18, 
                             "axes.titlesize":15, 
                             "axes.labelsize":13, 
                             "xtick.labelsize": 13,
                             "ytick.labelsize": 13,
                             "legend.fontsize": 9,
                             "legend.title_fontsize": 11}) 


# In[4]:


set_config(display='diagram')


# # EDA

# In[5]:


data = pd.read_csv('archive/pd_speech_features.csv')


# In[6]:


data.info()


# There are 756 rows with a lot of columns - 755. Each person has 3 records, so there are 252 patients overall

# In[7]:


data.head()


# From the dataset description, attributes are extracted using:
# 
# > Various speech signal processing algorithms including Time Frequency Features, Mel Frequency Cepstral Coefficients (MFCCs), Wavelet Transform based Features, Vocal Fold Features and TWQT features have been applied to the speech recordings of Parkinson's Disease (PD) patients to extract clinically useful information for PD assessment.
# 
# Without diving into the domain area, I cannot extract features better than the authors of the related paper.

# In[8]:


sizes = dict(data['class'].value_counts())

plt.figure(figsize=(12, 8))
plt.title("Does the person has Parkinson's Disease")
plt.pie(sizes.values(), labels=['Yes', 'No'], autopct="%.1f%%", pctdistance=0.85, colors=PALETTE)

plt.show()


# Target feature is unbalanced, like in most medical data, but this time we have 0 class underrepresented (no Parkinsons's Disease)

# I guess gender is important feature, because vocal features of male and female may vary a lot.  
# So let's look on gender proportions in each class

# In[9]:


sns.heatmap(pd.crosstab(data['class'], data['gender']).divide(3).astype('int64'), 
            yticklabels=['No PD', 'PD'],
            xticklabels=['Female', 'Male'],
            annot=True,
            fmt='d')
plt.title('Number of males and females in each class')
plt.show()


# We have:
# - 41 Females and 23 Males without PD
# - 81 Females and 107 Males with PD
# 
# Males are underrepresented in No PD group
# 
# Females are underrepresented in PD group

# __Spplitting the data__

# In[10]:


X = data.drop(columns='class')
y = data['class']


# Correlations in the dataset:

# In[11]:


corr_matr = X.drop(columns=['id', 'gender']).corr(method='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(corr_matr, cmap='coolwarm', square=True)
plt.title("Pearson's correlation heatmap on raw dataset")
plt.show()


# We know that there are correlated features in this dataset, so non-robust to multicollinearity models might suffer

# # Feature scaling

# If we look on feature distributions, we will see somewhere near to normal skewed distributions and hardly skewed distributions

# These are the first 20 features, but i checked features in each attribute type (see attribute descriptin) and the distributions are pretty similar

# In[12]:


g = sns.pairplot(data=X.iloc[:, 2:23], 
                 kind='scatter')

plt.tight_layout()


# I will use QuantileTransformer for feature scaling. This method transforms the features to follow a uniform or a normal distribution. This transformation tends to spread out the most frequent values. It also reduces the impact of outliers

# First two features are `id` and `gender`, we don't need to tranform them

# In[98]:


scaler = gen_features(
    columns = [[c] for c in X.columns.values if c not in ['gender', 'id']],
    classes=[{'class': QuantileTransformer, 'output_distribution': 'normal'}]
)


# I will scale all data, just to show how this scaling works

# In[100]:


scaling_mapper = DataFrameMapper(scaler, default=None, df_out=True)
X_scaled = scaling_mapper.fit_transform(X)


# Pairplot of features after scaling:

# In[101]:


g = sns.pairplot(data=X_scaled.iloc[:, 2:23], 
                 kind='scatter')

plt.tight_layout()


# In[17]:


corr_matr = X_scaled.drop(columns=['id', 'gender']).corr(method='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(corr_matr, cmap='coolwarm', square=True)
plt.title("Pearson's correlation heatmap on scaled dataset")
plt.show()


# And the correlations after scaling have become bigger (colors are more saturated)

# # Cross-validation scheme

# Cross-validation in our data set requires stratifying by `class` and also grouping by `id`. Records from one person can be very similar, so to prevent data leakage, we should group person's records

# In[102]:


def cross_validate(estimator, 
                   X: pd.DataFrame, 
                   y: pd.Series, 
                   print_fold_scores=False, 
                   plot_cm=False, 
                   scaling=False,
                   upsampling=False, 
                   resampling=False, 
                   pca=False) -> pd.DataFrame:
    """Calculates estimators's cross-validation scores on (X, y) dataset 
    
    Parameters
    ----------
    estimator : estimator to evaluate
    X : Data set to cross-validate on
    y : Data set target labels
    print_fold_scores : Set to True to print scores for each fold in cv
    plot_cm : Set to True to plot cofusion matrix
    scaling : DataFrameMapper with a scaler in it, used to scale X_train and X_test.
              If False - scaling is not used
    upsampling : Set to True to upsample train data in each fold
    resampling : Set to True to resample train data in each fold
    pca : Used as n_components parameter in PCA. If False - pca is not used 
    
    Returns
    -------
    mean_cv_scores_df : DataFrame with mean cross validation scores for estimator
    """
    # defining scores to evaluate
    cv_scores = {'Accuracy': [],
                 'Recall': [],
                 'Precision': [],
                 'F1': []}
    
    estimator_name = type(estimator).__name__
    
    # Stratify by target and group by id in order to prevent getting records 
    # of one person in train and test set
    fold = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE)
    
    for train_index, test_index in fold.split(X, y, groups=X['id']):
        X_train, X_test = X.iloc[train_index].drop(columns='id'), X.iloc[test_index].drop(columns='id')
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # transformations before training
        if scaling:
            scaling.fit(X_train)
            X_train = scaling.transform(X_train)
            X_test = scaling.transform(X_test)
        if resampling:
            X_train, y_train = resample_gender(X_train, y_train)
        if upsampling:
            X_train, y_train = upsample(X_train, y_train)
        if pca:
            X_train, X_test = perform_pca(X_train, X_test, explained_variance=pca)

        estimator.fit(X_train, y_train)

        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)
        
        cv_scores['Accuracy'].append(accuracy_score(y_test, predictions))
        cv_scores['Recall'].append(recall_score(y_test, predictions, pos_label=1))
        cv_scores['Precision'].append(precision_score(y_test, predictions, pos_label=1))
        cv_scores['F1'].append(f1_score(y_test, predictions, average='binary'))
    
    # prints scores for each fold if True
    if print_fold_scores:
        for item in cv_scores.items():
            print(item)
    
    mean_cv_scores = {k: np.mean(v) for k, v in cv_scores.items()}
    mean_cv_scores_df = pd.DataFrame.from_dict(data={estimator_name: mean_cv_scores.values()}, 
                                               orient='index', 
                                               columns=mean_cv_scores.keys())
    
    if plot_cm:
        plot_confusion_matrix(y_test, predictions, estimator_name)
        
    return mean_cv_scores_df

def plot_confusion_matrix(y_true: pd.Series, 
                          y_pred: pd.Series, 
                          estimator_name: str):
    """Plots confusion matrix for the last fold
    
    Parameters
    ----------
    y_true : True class labels
    y_pred : Predicted class labels
    """
    conf = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf,         
                yticklabels=['No PD', 'PD'],
                xticklabels=['No PD', 'PD'],
                annot=True,
                fmt='d')
    plt.title(f'{estimator_name} confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
def upsample(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Upsamples dataset with SOMTENC 
    
    Parameters
    ----------
    X : Data set to uspsample
    y : Data set class labels
    
    Returns
    -------
    X_upsampled : upsampled dataset
    y_upsampled : upsampled dataset class labels
    """
    smotenc = SMOTENC(categorical_features=[X.columns.get_loc("gender")], 
                    random_state=RANDOM_STATE, 
                    sampling_strategy=1)
    
    X_upsampled, y_upsampled = smotenc.fit_resample(X, y)
    
    return X_upsampled, y_upsampled
    
def resample_gender(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Resamples gender proportions in each class
    
    Parameters
    ----------
    X : Data set to resample
    y : Data set class labels
    
    Returns
    -------
    X_resampled : resampled dataset
    y_resampled : resampled dataset class labels
    """
    
    X_full = X.copy()
    X_full['class_'] = y

    # resampling gender proportions in 1 class
    df_majority_1 = X_full.query('class_ == 1 and gender == 1')
    df_minority_1 = X_full.query('class_ == 1 and gender == 0')

    df_minority_resampled_1 = resample(df_minority_1, 
                                       replace=True,
                                       n_samples=len(df_majority_1),
                                       random_state=RANDOM_STATE)

    df_resampled_1 = pd.concat([df_majority_1, df_minority_resampled_1])

    # resampling gender proportions in 0 class
    df_majority_0 = X_full.query('class_ == 0 and gender == 0')
    df_minority_0 = X_full.query('class_ == 0 and gender == 1')

    df_minority_resampled_0 = resample(df_minority_0, 
                                       replace=True,
                                       n_samples=len(df_majority_0),
                                       random_state=RANDOM_STATE)

    df_resampled_0 = pd.concat([df_majority_0, df_minority_resampled_0])

    # Combining two resampled subsets
    df_resampled = pd.concat([df_resampled_1, df_resampled_0], ignore_index=True)

    X_resampled = df_resampled.drop(columns='class_')
    y_resampled = df_resampled['class_']
    
    return X_resampled, y_resampled

def perform_pca(X_train, X_test, explained_variance) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs PCA on X_train and X_test
    
    Parameters
    ----------
    X_train : Train data set to fit PCA
    X_test : Test data set to tranform with PCA  
    
    Returns
    -------
    X_train_pca : PCA-transformed train data set 
    y_test_pca : PCA-transformed test data set
    """
    pca = PCA(n_components=explained_variance).fit(X_train.drop(columns='gender'))
    pca_train_data = pca.transform(X_train.drop(columns='gender'))
    pca_test_data = pca.transform(X_test.drop(columns='gender'))
    
    X_train_pca = pd.DataFrame.from_records(data=pca_train_data)
    
    #reset index to map id and gender to pca data
    X_train.reset_index(inplace=True)
    X_train_pca['gender'] = X_train['gender']
    
    X_test_pca = pd.DataFrame.from_records(data=pca_test_data)
    
    #reset index to map id and gender to pca data
    X_test.reset_index(inplace=True)
    X_test_pca['gender'] = X_test['gender']
    
    return X_train_pca, X_test_pca

def display_side_by_side(dfs: list, titles: list):
    """Displays dataframes side by side
    
    Parameters
    ----------
    dfs : list of pandas.DataFrame
    titles : list of dataframe titles
    """
    output = ""
    combined = dict(zip(titles, dfs))
    for title, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(title)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))


# Now let's check the scores for diferent models out-of-box

# ## KNN

# In[103]:


models_results = cross_validate(KNeighborsClassifier(), X, y, scaling=scaling_mapper, plot_cm=True)


# In[104]:


models_results


# ## LogReg

# In[49]:


lg_cv = cross_validate(LogisticRegression(random_state=RANDOM_STATE), X, y, scaling=scaling_mapper, plot_cm=True)


# In[50]:


lg_cv


# In[51]:


models_results = models_results.append(lg_cv)


# ## DT

# In[27]:


dt_cv = cross_validate(DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6), X, y, plot_cm=True)


# In[28]:


dt_cv


# In[29]:


models_results = models_results.append(dt_cv)


# ## RF

# In[30]:


rf_cv = cross_validate(RandomForestClassifier(random_state=RANDOM_STATE, max_depth=6), X, y, plot_cm=True)


# In[31]:


rf_cv


# In[32]:


models_results = models_results.append(rf_cv)


# ## CatBoost

# In[33]:


catboost_cv = cross_validate(CatBoostClassifier(depth=6, cat_features=['gender'], verbose=False, random_seed=RANDOM_STATE), X, y, plot_cm=True)


# In[34]:


catboost_cv


# In[35]:


models_results = models_results.append(catboost_cv)


# ## LightGBM

# In[36]:


lgbm_cv = cross_validate(LGBMClassifier(max_depth=6, random_state=RANDOM_STATE), X, y, plot_cm=True)


# In[37]:


lgbm_cv


# In[38]:


models_results = models_results.append(lgbm_cv)


# ## XGBoost

# In[39]:


xgb_cv = cross_validate(XGBClassifier(max_depth=6, random_state=RANDOM_STATE, verbosity=0), X, y, plot_cm=True)


# In[40]:


xgb_cv


# In[41]:


models_results = models_results.append(xgb_cv)


# __Models comparison:__

# In[55]:


models_results.sort_values(by='F1', ascending=False, inplace=True)


# In[56]:


models_results


# LGBMClassifier gives the best Recall - Precision ratio and the best accuracy out-of-box

# # Upsampling with SMOTE

# Now let's try to cross validate with SMOTE upsampling

# In[43]:


setting = {
    'upsampling': True,
    'X': X,
    'y': y
}

models = [
    dict({'estimator': KNeighborsClassifier(),
          'scaling': scaling_mapper}, 
         **setting),
    dict({'estimator': LogisticRegression(random_state=RANDOM_STATE),
          'scaling': scaling_mapper}, 
         **setting),
    dict({'estimator': DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': RandomForestClassifier(max_depth=6, random_state=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': CatBoostClassifier(depth=6, cat_features=['gender'], verbose=False, random_seed=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': LGBMClassifier(max_depth=6, random_state=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': XGBClassifier(max_depth=6, verbosity=0, random_state=RANDOM_STATE)}, 
         **setting)
]


# In[57]:


models_results_upsampling = pd.DataFrame()
for model in models:
    models_results_upsampling = models_results_upsampling.append(cross_validate(**model))
    
models_results_upsampling.sort_values(by='F1', ascending=False, inplace=True)


# In[58]:


display_side_by_side([models_results, models_results_upsampling], 
                     titles=['original data cv scores', 'upsampled data cv scores'])


# As expected, recall decreased, precision increased, that is not really what we want

# # Resampling 

# This is how resampling method works:

# In[59]:


X_resampled, y_resampled = resample_gender(X, y)

sns.heatmap(pd.crosstab(y_resampled, X_resampled['gender']).divide(3).astype('int64'), 
            yticklabels=['No PD', 'PD'],
            xticklabels=['Female', 'Male'],
            annot=True,
            fmt='d')

plt.title('Number of males and females in each class after resampling')
plt.show()


# In[60]:


del X_resampled, y_resampled


# Now let's cross-validate on resampled data, so gender proportion in each class are equal

# In[61]:


setting = {
    'resampling': True,
    'X': X,
    'y': y
}

models = [
    dict({'estimator': KNeighborsClassifier(),
          'scaling': scaling_mapper}, 
         **setting),
    dict({'estimator': LogisticRegression(random_state=RANDOM_STATE),
          'scaling': scaling_mapper}, 
         **setting),
    dict({'estimator': DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': RandomForestClassifier(max_depth=6, random_state=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': CatBoostClassifier(depth=6, cat_features=['gender'], verbose=False, random_seed=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': LGBMClassifier(max_depth=6, random_state=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': XGBClassifier(max_depth=6, verbosity=0, random_state=RANDOM_STATE)}, 
         **setting)
]


# In[62]:


models_results_resampling = pd.DataFrame()
for model in models:
    models_results_resampling = models_results_resampling.append(cross_validate(**model))
    
models_results_resampling.sort_values(by='F1', ascending=False, inplace=True)


# In[65]:


display_side_by_side([models_results, models_results_resampling], 
                     titles=['original data cv scores', 'resampled data cv scores'])


# Seems like the scores are a little lower than on raw data.

# __CV scores on original data, upsampled data and resampled data compared:__

# In[66]:


display_side_by_side([models_results, models_results_upsampling, models_results_resampling], 
                     titles=['original data cv scores', 'upsampled data cv scores', 'resampled data cv scores'])


# # Dimensionality Reduction

# ## PCA

# Let's look how the whole data is distributed in 3 dimensions (using PCA)

# In[72]:


pca_data = PCA(n_components=3).fit_transform(X_scaled.drop(columns=['id', 'gender']))
plot_df = pd.DataFrame.from_records(data=pca_data,columns=['pc1','pc2', 'pc3'])
plot_df['target'] = y
fig = px.scatter_3d(plot_df, x='pc1', y='pc2', z='pc3', color='target', width=800, height=800)
fig.show()


# As we see, the data is not very separable even on 3 dimensions.

# Let's find the optimal number of components

# In[73]:


EXPLAINED_VARIANCE = 0.99

pca = PCA(n_components=EXPLAINED_VARIANCE).fit(X_scaled.drop(columns=['id', 'gender']))


# In[74]:


plt.figure(figsize=(15, 10))

plt.bar(range(len(pca.explained_variance_)), pca.explained_variance_ratio_, align='center',
        label='Component explained variance ratio', edgecolor = "none")
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component')
plt.title('Explained variance ratio for each principal component')
plt.legend()
plt.tight_layout()


# In[75]:


n_components = len(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(24, 8))
x_ticks = np.arange(1, n_components + 1, step=1)
y_values = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(x_ticks, y_values, marker='.', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, n_components + 1, step=10))
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=EXPLAINED_VARIANCE, color='r', linestyle='-')

plt.axvline(x=n_components, color='r', linestyle='--')

plt.text(0.5, 1.01, f'{EXPLAINED_VARIANCE*100}% threshold', color = 'red')
plt.text(n_components + 1, 0.1, f'{n_components}', color = 'red')

ax.grid(axis='x')
plt.xticks(rotation=0)
plt.show()


# I would choose 150 number of components, that's  5 times less features, but they still explain most of the variance (around 95%)

# In[76]:


pca.explained_variance_ratio_[:150].sum()


# That's how `perform_pca` method works on our data (just an example to see what's the output of PCA):

# In[79]:


train_pca, test_pca = perform_pca(X_scaled.drop(columns=['id'])[:600], X_scaled.drop(columns=['id'])[600:], 150)


# In[81]:


test_pca


# Let's check how PCA affects our models. This time, even trees models are trained on scaled data, because we must scale the data before PCA

# In[82]:


setting = {
    'pca': 150,
    'X': X_scaled,
    'y': y
}

models = [
    dict({'estimator': KNeighborsClassifier(),
          'scaling': scaling_mapper}, 
         **setting),
    dict({'estimator': LogisticRegression(random_state=RANDOM_STATE),
          'scaling': scaling_mapper}, 
         **setting),
    dict({'estimator': DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': RandomForestClassifier(max_depth=7, random_state=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': CatBoostClassifier(depth=6, verbose=False, random_seed=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': LGBMClassifier(max_depth=6, random_state=RANDOM_STATE)}, 
         **setting),
    dict({'estimator': XGBClassifier(max_depth=6, verbosity=0, random_state=RANDOM_STATE)}, 
         **setting)
]


# In[84]:


models_results_pca = pd.DataFrame()
for model in models:
    models_results_pca = models_results_pca.append(cross_validate(**model))
    
models_results_pca.sort_values(by='F1', ascending=False, inplace=True)


# In[85]:


display_side_by_side([models_results, models_results_pca], 
                     titles=['original data cv scores', 'pca data cv scores'])


# So the results are not good, PCA negatively affects F1 score and accuracy overall. Some models are definetly overfitted (RFC, CatBoost)

# As the result model I would choose LGBMClassifier with resampling.
# - LGBMClassifier provides us with the best recall/precision tradeoff out-of-box
# - LGBM learning is relatively fast (in comparison with other boostings), so the tuning will not take a lot of time and we can experiment more with that
# - The intuition about resampling gender proportions in each class: training on resampled data will be more robust to test sets, that not look like train set (in terms of class/gender proportions). And i assume that the model will be used equally on males and females

# # Model tuning

# ## LGBMClassifier

# As long as I use grouping, resampling and stratifying, I have to write my own wrapper transformer with `fit_resample` method

# In[99]:


class CustomResamplingTransformer():
    
    def fit_resample(self, X, y):
        return resample_gender(X, y)


# In[100]:


pipeline = Pipeline([
    ('resample', CustomResamplingTransformer()),
    ('estimator', LGBMClassifier(random_state=RANDOM_STATE))
])


# I will use F1 score in GridSearch, because it takes into account both Recall and Precision for 1 class.  
# I do not use first class Recall for tuning, because the model will just classify almost all objects as 1 and that is a bad model

# In[101]:


params_1 = {
    'estimator__num_leaves':[10, 20, 30, 40, 60, 80, 100],
    'estimator__n_estimators': [200, 250, 300, 350],
    'estimator__max_depth':[-1, 4, 6, 8, 10, 15]}


# In[102]:


gs_1 = GridSearchCV(pipeline,
                    param_grid=params_1,
                    cv=StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X, y, groups=X['id']),
                    scoring='f1')


# In[103]:


gs_1.fit(X.drop(columns='id'), y)


# In[105]:


gs_1.best_score_


# In[106]:


gs_1.best_params_


# In[122]:


def extract_estimator_params(gs_params: dict) -> dict:
    '''Extracts estimator params from GridSearchCV best_params_
    
    Parameters
    ----------
    gs_params : GridSearchCV best_params_ attribute
    
    Returns
    -------
    estimator_params : dict, containing estimator's params
    '''
    estimator_params = {k.split('__')[-1]: v for k, v in gs_params.items()}
    return estimator_params


# In[114]:


estimator_1_params = extract_estimator_params(gs_1.best_params_)


# In[115]:


estimator_1_params


# In[116]:


cross_validate(LGBMClassifier(random_state=RANDOM_STATE, **estimator_1_params), X, y, resampling=True, print_fold_scores=True, plot_cm=True)


# Both precision and recall has increased
# 
# We can also see that, for example, the third and the last fold scores differs a lot. And that is happend because of small dataset, i guess 

# The second lap of GridSearch (now we specify parameters in smaller limits):

# In[117]:


params_2 = {
    'estimator__num_leaves':[16, 18, 20, 22, 24],
    'estimator__n_estimators': [330, 340, 350, 360, 370],
    'estimator__max_depth':[3, 4, 5]}


# In[118]:


gs_2 = GridSearchCV(pipeline,
                    param_grid=params_2,
                    cv=StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X, y, groups=X['id']),
                    scoring='f1')


# In[119]:


gs_2.fit(X.drop(columns='id'), y)


# In[120]:


gs_2.best_score_


# In[121]:


gs_2.best_params_


# In[122]:


estimator_2_params = extract_estimator_params(gs_2.best_params_)


# In[123]:


cross_validate(LGBMClassifier(random_state=RANDOM_STATE, **estimator_2_params), X, y, print_fold_scores=True, plot_cm=True, resampling=True)


# __The third lap of GridSearch:__

# In[124]:


params_3 = {
    'estimator__num_leaves':[20],
    'estimator__n_estimators': [315, 320, 325, 330, 335],
    'estimator__max_depth':[5]}


# In[125]:


gs_3 = GridSearchCV(pipeline,
                    param_grid=params_3,
                    cv=StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X, y, groups=X['id']),
                    scoring='f1')


# In[126]:


gs_3.fit(X.drop(columns='id'), y)


# In[127]:


gs_3.best_score_


# In[128]:


gs_3.best_params_


# In[130]:


estimator_3_params = extract_estimator_params(gs_3.best_params_)


# In[131]:


cross_validate(LGBMClassifier(random_state=RANDOM_STATE, **estimator_3_params), X, y, print_fold_scores=True, plot_cm=True, resampling=True)


# The scores didn't change, so the final params are

# In[132]:


estimator_3_params


# ## Logistic Regression

# Let's try to tune Logistic Rgression to see how such a simple model can perform compared to more complex model like LGBM

# I will remove resampling from pipeline

# In[90]:


pipeline_logreg = Pipeline([
    ('scaler', scaling_mapper),
    ('estimator', LogisticRegression(random_state=RANDOM_STATE))
])


# In[95]:


weights_for_0_class = np.linspace(0.5, 0.6, 20)

params_logreg = [
    {'estimator__penalty' : ['l1', 'l2', 'none', 'elasticnet'],
     'estimator__C' : np.linspace(-0.002, 0.002, 10),
     'estimator__class_weight': ['none'] + [{0:x, 1:1.0-x} for x in weights_for_0_class]}
]


# In[116]:


gs_logreg = GridSearchCV(pipeline_logreg,
                         param_grid=params_logreg,
                         cv=StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X, y, groups=X['id']),
                         scoring='f1')


# In[117]:


gs_logreg.fit(X.drop(columns='id'), y)


# In[118]:


gs_logreg.best_score_


# In[119]:


gs_logreg.best_params_


# In[123]:


logreg_params = extract_estimator_params(gs_logreg.best_params_)


# In[126]:


cross_validate(LogisticRegression(random_state=RANDOM_STATE, **logreg_params), X, y, scaling=scaling_mapper, print_fold_scores=True, plot_cm=True)


# That's a nice score for a LogisticRegression. But we see that it is less consistent on different folds in comparison to LGBM

# # Results

# What has been done in this work:
# - Simple EDA (features are already extracted)
# - Custom cross-validation scheme with stratifying by `class` and grouping by `id` 
# - Out-of-box models comparison (LGBMClassifier is the best)
# - Upsampling using SMOTENC (badly affects the scores)
# - Resampling gender proportions in each class (used in the result model) 
# - PCA (badly affects the scores)
# - Model tuning (LGBM and LogisticRegression)

# So we have the following model

# LGBMClassifier on resampled data with the parameters:
# - max_depth: 5
# - n_estimators: 320
# - num_leaves: 20

# And the mean cv scores of this model are:
# - Accuracy: 0.858719
# - Recall: 0.959212
# - Precision: 0.865519
# - F1: 0.909701
