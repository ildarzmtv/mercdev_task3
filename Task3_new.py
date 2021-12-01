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

# ![download.png](attachment:0ccfa793-1d29-4308-94da-fa52a1845591.png)

# # Import required libraries

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn import set_config
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE

from imblearn.pipeline import Pipeline

from IPython.display import HTML, display

from mrmr import mrmr_classif

import warnings
warnings.filterwarnings("ignore")

sns.set_theme()


# In[3]:


RANDOM_STATE = 42


# In[4]:


sns.set_context("paper", rc={"font.size":12, 
                             "figure.titlesize":18, 
                             "axes.titlesize":15, 
                             "axes.labelsize":13, 
                             "xtick.labelsize": 13,
                             "ytick.labelsize": 13,
                             "legend.fontsize": 9,
                             "legend.title_fontsize": 11}) 


# In[5]:


set_config(display='diagram')


# # EDA

# In[6]:


data = pd.read_csv('archive/pd_speech_features.csv')


# In[7]:


data.info()


# There are 756 rows with a lot of columns - 755. Each person has 3 records, so there are 252 patients overall (130 men and 122 women)

# ## Target variable

# In[8]:


sizes = dict(data['class'].value_counts())

plt.figure(figsize=(12, 8))
plt.title("Does the person has Parkinson's Disease")
plt.pie(sizes.values(), labels=['Yes', 'No'], autopct="%.1f%%", pctdistance=0.85)

plt.show()


# Target feature is unbalanced, like in most medical data, but this time we have 0 class underrepresented (no Parkinsons's Disease)

# I'll create lists of columns names according to the attribute groups from dataset description

# - Baseline Features: Col3 to Col23
# - Intensity Parameters: Col24 to Col26
# - Formant Frequencies: Col27 to Col30
# - Bandwidth Parameters: Col31 to Col34
# - Vocal Fold: Col35 to Col56
# - MFCC: Col57 to Col140
# - Wavelet Features: Col141 to Col322
# - TQWT Features: Col323 to Col754
# - Class: Col755

# In[9]:


BASELINE = list(data.iloc[:, 2:23].columns.values)
INTENSITY = list(data.iloc[:, 23:26].columns.values)
FORMANT_FREQUENCIES = list(data.iloc[:, 26:30].columns.values)
BANDWIDTH = list(data.iloc[:, 30:34].columns.values)
VOCAL_FOLD = list(data.iloc[:, 34:56].columns.values)
MFCC = list(data.iloc[:, 56:140].columns.values)
WAVELET = list(data.iloc[:, 140:322].columns.values)
TQWT = list(data.iloc[:, 322:754].columns.values)


# ## Correlations

# In[10]:


def plot_correlation_matrix(data, title):
    corr_matr = data.corr(method='pearson')
    plt.figure(figsize=(10,10))
    sns.heatmap(corr_matr, cmap='coolwarm', square=True)
    plt.title(title)
    plt.show()


# In[11]:


plot_correlation_matrix(data, f" Pearson's correlations")


# We see some correlation patterns in different areas on the heatmap. We will look on them more closely later 

# __Correlations with the target__

# It’s not completely correct to calculate the correlation coefficient in binary classification problem using __Pearson's__ method. For a perfect predictor, we expect a Pearson coefficient absolute value equal to 1, but we could not achieve this value if we have one binary feature. It’s not important, however. We are using Pearson correlation coefficient to __sort our features__ from the most relevant to the least one, so as long as the coefficient calculation is the same, we can compare the features between them.

# In[12]:


def check_attribute_group(str: value):
    """Checks the name of feature and returns the corresponding attribute group
    
    Parameters
    ----------
    value : Name of feature
    """
    if value in BASELINE:
        return 'BASELINE'
    elif value in INTENSITY:
        return 'INTENSITY'
    elif value in FORMANT_FREQUENCIES:
        return 'FORMANT_FREQUENCIES'
    elif value in BANDWIDTH:
        return 'BANDWIDTH'
    elif value in VOCAL_FOLD:
        return 'VOCAL_FOLD'
    elif value in MFCC:
        return 'MFCC'
    elif value in WAVELET:
        return 'WAVELET'
    elif value in TQWT:
        return 'TQWT'
    else: return 'GENERAL'


# In[13]:


target_correlations = pd.DataFrame(data.drop(columns='class').corrwith(data['class']).apply(np.abs).sort_values(ascending=False), columns=["Pearson's correlation"])
target_correlations['Attribute group'] = target_correlations.index.map(check_attribute_group)


# In[14]:


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


# Top 100 features by Pearson's correlation:

# In[15]:


display_side_by_side([target_correlations.nlargest(100, columns="Pearson's correlation").iloc[:50],
                      target_correlations.nlargest(100, columns="Pearson's correlation").iloc[50:]],
                     ['Top 1-50', 'Top 51-100'])


# Let's see which attribute group is represented the most in top 100 features

# In[16]:


target_correlations.nlargest(100, columns="Pearson's correlation")['Attribute group'].value_counts()


# So there are mostly `TQWT`, `MFCC`, `WAVELET` attribute groups in top 100 features
# This prompts us to pay special attention to these attribute groups.

# Let's try to explore data by attribute groups

# ## Baseline features

# In[17]:


features = BASELINE
plot_correlation_matrix(data[features + ['class']], f" Pearson's correlations")


# In[18]:


data[BASELINE].head(10)


# The most correlated features:
# - `numPulses`, `numPeriodsPulses`, `meanPeriodPulses`: Number of glottal pulses and number of periods are obviously correlated, and the mean period of pulse is calculated as *SoomeRecordTime* / *NumOfPeriods*, so the correlation with those two is negative
# - Different varians of `shimmer`
# - Different variants of `jitter`
# - Noise-to-harmonics and harmonics-to-noise ratios are negatively correlated, from the name of this variables i expect the correlation to be $NHR=\frac{1}{HNR}$. 

# ## Intensity parameters  

# In[19]:


features = INTENSITY
plot_correlation_matrix(data[features + ['class']], f" Pearson's correlations")


# All three intensity parameters are correlated and that is obvious. Mean and Max are more correlated than Mean and Min. I can guess that speech intensity distribution is skewed towards max value

# ## MFCC

# In[20]:


features = MFCC
plot_correlation_matrix(data[features + ['class']], f" Pearson's correlations")


# There are correlations in the lower right corner.  
# 
# > In this study, mean and standard
# deviation of the original 13 MFCCS plus log-energy of the signal and
# their first–second derivatives are employed as features resulting in
# 84 features 
# 
# Standard deviations of Mel Frequency Cepstral Coefficient, it's delta and delta-delta are correlated with each other, while means of those values are not correlated. Absolute values can vary a lot, so the means may not correlate, but variance and standard deviation of those samples may correlate

# Let's look on this sets of 13 coefficients more closely.  
# 
# 
# 
# 

# Description of the graph below:
# 
# On the x axis we have 13 points, which represent 13 MFCCS features  
# On the y axis we have the value of each MFCC feature
# 
# Each point on the plot is the median of this MFCC feature. And all this points are connected with the line  
# Orange - PD class, Blue - No PD class
# 
# The filling around the dots represents quantiles of 25 and 75 respectively, so we can see the intersections in distribution of this features for two classes
# 
# There are also boxplots for this features for each class to see the tails and skewness of distributions

# In[21]:


# start from 0, max - 5
# set 1, bias 1; set 2, bias 3; set 3, bias 4; set 4, bias 5; set 5, bias 6;
feature_set = 0
bias = 1
feature_set_size = 13
features = MFCC

class_0_data = data[features + ['class']].loc[data['class'] == 0].iloc[:, feature_set*feature_set_size + bias: (feature_set+1)*feature_set_size + bias]
class_1_data = data[features + ['class']].loc[data['class'] == 1].iloc[:, feature_set*feature_set_size + bias: (feature_set+1)*feature_set_size + bias]


fig, axes = plt.subplots(3, 1, figsize=(15, 21))

sns.lineplot(x=range(feature_set_size), y=class_0_data.median(), ax=axes[0])
sns.lineplot(x=range(feature_set_size), y=class_1_data.median(), ax=axes[0])
axes[0].set_xticks(range(feature_set_size))
axes[0].set_title(f"{class_0_data.columns[0]} - {class_0_data.columns[-1]} median values")

sns.boxplot(data=class_0_data, ax=axes[1])
axes[1].set_xticklabels(range(feature_set_size))
axes[1].set_title(f"{class_0_data.columns[0]} - {class_0_data.columns[-1]} boxplots for NO PD class")

sns.boxplot(data=class_1_data, ax=axes[2])
axes[2].set_xticklabels(range(feature_set_size))
axes[2].set_title(f"{class_0_data.columns[0]} - {class_0_data.columns[-1]} mean values for PD class")

quartiles1_class0 = class_0_data.quantile(.25)
quartiles3_class0 = class_0_data.quantile(.75)

quartiles1_class1 = class_1_data.quantile(.25)
quartiles3_class1 = class_1_data.quantile(.75)

axes[0].fill_between(range(feature_set_size), quartiles1_class0, quartiles3_class0, alpha=0.2);
axes[0].fill_between(range(feature_set_size), quartiles1_class1, quartiles3_class1, alpha=0.2); 

plt.show()


# `mean_MFCC_2nd_coef` is in top of correlations and from this plot we can see that interception of orange and blue filling is relatively small, so that feature is more separable for two classes than others

# Same intuition can be applied to `std_delta_delta` features, which are also in top of correlations ![image.png](attachment:25adb994-5a11-4cee-a868-2d8a290c1f78.png)

# ## Wavelet

# >  In our study, to quantify the performance of WT based features, which are
# obtained from the raw F0 contour and also from the log transform
# of the F0 contour, 10-levels discrete wavelet
# transform is applied to speech samples. After decomposition, the
# energy, Shannon’s and the log energy entropy, and the Teager–
# Kaiser energy of both the approximation and detailed coefficients
# are calculated resulting in 182 WT features related with F0

# In[22]:


features = WAVELET
plot_correlation_matrix(data[features + ['class']], f" Pearson's correlations")


# Can't  really explain the correlations without proper knowledge in signal processing, but from wavelet features there are some good predictors, that are also in top-100
# ![image.png](attachment:f49595dd-bc49-4af3-8898-cd9fabc90ca3.png)![image.png](attachment:ee8b8141-21f5-4c2c-a57f-7473275d2201.png)![image.png](attachment:6922e4f4-6f92-4974-8671-3c578ec05ba9.png)

# ## TQWT

# > In this study, we apply, to the best of our knowledge for the first time, the tunable Q-factor wavelet transform
# (TQWT) to the voice signals of PD patients for feature extraction, which has higher frequency resolution
# than the classical discrete wavelet transform

# Features of this type are obtained from raw F0, using TQWT with 36 levels of decomposition. And we see sets of 36 features that are correlated with each other 

# In[23]:


features = TQWT
plot_correlation_matrix(data[features + ['class']], f" Pearson's correlations")


# 11 and 12 levels of decompositions are in top correlated features, let's look on them:

# In[26]:


# start from 0, max - 11
feature_set = 7
bias = 0
feature_set_size = 36
features = TQWT

class_0_data = data[features + ['class']].loc[data['class'] == 0].iloc[:, feature_set*feature_set_size + bias: (feature_set+1)*feature_set_size + bias]
class_1_data = data[features + ['class']].loc[data['class'] == 1].iloc[:, feature_set*feature_set_size + bias: (feature_set+1)*feature_set_size + bias]


fig, axes = plt.subplots(3, 1, figsize=(15, 21))

sns.lineplot(x=range(feature_set_size), y=class_0_data.median(), ax=axes[0])
sns.lineplot(x=range(feature_set_size), y=class_1_data.median(), ax=axes[0])
axes[0].set_xticks(range(feature_set_size))
axes[0].set_title(f"{class_0_data.columns[0]} - {class_0_data.columns[-1]} median values")

sns.boxplot(data=class_0_data, ax=axes[1])
axes[1].set_xticklabels(range(feature_set_size))
axes[1].set_title(f"{class_0_data.columns[0]} - {class_0_data.columns[-1]} boxplots for NO PD class")

sns.boxplot(data=class_1_data, ax=axes[2])
axes[2].set_xticklabels(range(feature_set_size))
axes[2].set_title(f"{class_0_data.columns[0]} - {class_0_data.columns[-1]} mean values for PD class")

quartiles1_class0 = class_0_data.quantile(.25)
quartiles3_class0 = class_0_data.quantile(.75)

quartiles1_class1 = class_1_data.quantile(.25)
quartiles3_class1 = class_1_data.quantile(.75)

axes[0].fill_between(range(feature_set_size), quartiles1_class0, quartiles3_class0, alpha=0.2);
axes[0].fill_between(range(feature_set_size), quartiles1_class1, quartiles3_class1, alpha=0.2); 

plt.show()


# And intersection of filled area is small for this features, that's why they are in top. We can also observe skewness in such types of features

# Another examples:

# ![image.png](attachment:d78d9e67-1bdd-4594-9746-fd0368ea64ab.png) ![image.png](attachment:db1f6287-934c-48a4-b1b7-b99b66c3c4eb.png)

# # Splitting the data

# In[27]:


X = data.drop(columns='class')
y = data['class']


# I would use train/test split for the start to check if the model is overfitting or not
# 
# While splitting the data into train and test set, we should stratify subsamples by `class` (because the data is umbalanced) and also group by `id`. Records from one person can be very similar, so to prevent data leakage, we should group person's records
# 
# I think 80% of the data is enough for the model to be trained, so the test is 20%

# In[28]:


train_indicies, test_indicies = next(StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X, y, groups=X['id']))


# In[29]:


X_train, X_test = X.iloc[train_indicies], X.iloc[test_indicies]
y_train, y_test = y.iloc[train_indicies], y.iloc[test_indicies]


# # Scaling

# From the EDA step we can see, that the most correlated features are long-tailed, and extreme values might be considered as "outliers". I would try 2 approaches: outlier-robust scaling and non-robust scaling
# 
# For non-robust methods i will try: StandardScaler, MinMaxScaler  
# For robust methods: QuantileTransformer, RobustScaler
# 

# Some helper functions:

# In[30]:


def get_scores(name, y_train, y_pred_train, y_test, y_pred_test):
    """Calculates accuracy, recall, precision, f1 scores for train and test sets
    
    Parameters
    ----------
    name : The name of the model (it will be shown in the resulting dataframe)
    X_train : Train set
    y_train : Train set labels
    X_test : Test set
    y_test : Test set labels
    
    Returns
    -------
    Function returns the dataframe with accuracy, recall, precision and f1 scores 
    for train and test sets
    """
    scores_train = {}
    scores_train['Accuracy'] = accuracy_score(y_train, y_pred_train)
    scores_train['Recall'] = recall_score(y_train, y_pred_train, pos_label=1)
    scores_train['Precision'] = precision_score(y_train, y_pred_train, pos_label=1)
    scores_train['F1'] = f1_score(y_train, y_pred_train, average='binary')
    
    scores_test = {}
    scores_test['Accuracy'] = accuracy_score(y_test, y_pred_test)
    scores_test['Recall'] = recall_score(y_test, y_pred_test, pos_label=1)
    scores_test['Precision'] = precision_score(y_test, y_pred_test, pos_label=1)
    scores_test['F1'] = f1_score(y_test, y_pred_test, average='binary')
    
    index = pd.MultiIndex.from_tuples([(name, 'Train'), (name, 'Test')], names=["Method", "Subsample"])
    scores_df = pd.DataFrame(data=[scores_train.values(), scores_test.values()], 
                             index=index, 
                             columns=scores_test.keys())
    return scores_df

def evaluate_model(model, 
                   str: name, 
                   pd.DataFrame: X_train, 
                   pd.Series: y_train, 
                   pd.DataFrame: X_test, 
                   pd.Series: y_test):
    """Evaluates model's performance on train and test data by printing 
    classification report and plotting confusion matrix
    
    Parameters
    ----------
    model : The model to evaluate
    X_train : Train set
    y_train : Train set labels
    X_test : Test set
    y_test : Test set labels
    
    Returns
    -------
    Function returns the dataframe with accuracy, recall, precision and f1 scores 
    for train and test sets
    """
    
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print("Train:")
    print(classification_report(y_train, y_pred_train))

    print("Validation:")
    print(classification_report(y_test, y_pred_test))
    
    plot_confusion_matrix(y_test, y_pred_test)
    
    return get_scores(name, y_train, y_pred_train, y_test, y_pred_test)

def plot_confusion_matrix(y_true: pd.Series, 
                          y_pred: pd.Series):
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
    plt.title(f'LogisticRegression confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Every chosen scaler method was passed through a following flow (example for standard scaler):  
# 
# GridSearch uses 5 folds on training set, so 20% of our 80% train sample is used for validation 

# In[31]:


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=RANDOM_STATE))
])


# In[32]:


weights_for_0_class = np.linspace(0.5, 0.6, 20)

params_logreg = [
    {'clf__penalty' : ['l1', 'l2', 'none', 'elasticnet'],
     'clf__C' : np.linspace(0.002, 0.003, 20),
     'clf__class_weight': ['none'] + [{0:x, 1:1.0-x} for x in weights_for_0_class]}
]

gs_logreg = GridSearchCV(pipeline,
                         param_grid=params_logreg,
                         cv=StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X_train, y_train, groups=X_train['id']),
                         scoring='f1',
                         n_jobs=-1)


# In[33]:


gs_logreg.fit(X_train.drop(columns='id'), y_train)


# In[34]:


gs_logreg.best_params_


# In[35]:


gs_logreg.best_estimator_


# In[38]:


scaler_results = evaluate_model(gs_logreg.best_estimator_, 'StandardScaler', X_train.drop(columns='id'), y_train, X_test.drop(columns='id'), y_test)


# In[39]:


scaler_results


# __Comaprison of scaling methods:__ 

# In[668]:


scaling_comparison


# And the `StandardScaler` works better than others. What's interesting is that more robust methods (RobustScaler and QuantileTransformer) provide us with more overfitting, the gap between train and test scores is bigger.  
# This model will be considered as the baseline model, and since it is not really overfitted, we can stay with that train/test split further

# # Feature selection

# Our dataset consists of 755 features, and there are definetly redundant ones. Some of them are correlated with each other, others are just not good predictors.  
# 
# To reduce the dimensionality, I'll try 2 feature selection methods - Recursive Feature Elimination (RFE) and Maximum Relevance — Minimum Redundancy (MRMR).  
# RFE is popular because it is easy to configure and use and because it is effective at selecting the most relevantfeatures  in predicting the target variable.  
# The MRMR approach is based on maximizing the joint dependency of top ranking variables on the target variable by reducing the redundancy among them. [MRMR explanation]("https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b") 

# For each feature selection method i tuned models with 50, 100, 150, 200, 250 features and that is the results:

# In[792]:


display_side_by_side([mrmr_comparison, rfe_comparison], 
                     titles=['MRMR', 'RFE'])


# For MRMR method i would choose 150 features as the optimal number, because the Recall is raised, while Precision is still pretty good.  
# For RFE, 150 features is also the best choice in my opinion, because the Recall and F1 both increased
# 
# As for overall performance, RFE with 150 features is the best choice 

# # Final model

# Our final model consists of feature scaling with StandardScaler, feature selection with RFE (150 features) and LogisticRegression

# In[40]:


model = gs_logreg.best_estimator_
rfe = RFE(estimator=model, n_features_to_select=150, importance_getter='named_steps.clf.coef_')
rfe.fit(X_train.drop(columns='id'), y_train)
selected_features = rfe.get_feature_names_out()


# In[41]:


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=RANDOM_STATE))
])

weights_for_0_class = np.linspace(0.4, 0.8, 30)

params_logreg = [
    {'clf__penalty' : ['l1', 'l2', 'none', 'elasticnet'],
     'clf__C' : np.linspace(0.002, 0.004, 30),
     'clf__class_weight': ['none'] + [{0:x, 1:1.0-x} for x in weights_for_0_class]}
]

gs_logreg_fs = GridSearchCV(pipeline,
                            param_grid=params_logreg,
                            cv=StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X_train, y_train, groups=X_train['id']),
                            scoring='f1',
                            n_jobs=-1)

gs_logreg_fs.fit(X_train[selected_features], y_train)


# In[42]:


gs_logreg_fs.best_params_


# In[43]:


rfe_score = evaluate_model(gs_logreg_fs.best_estimator_, f'RFE (n=150)', X_train[selected_features], y_train, X_test[selected_features], y_test)


# In[86]:


rfe_score


# Now let's see which features are presented in selected ones

# In[44]:


attribute_types = np.array(list(map(check_attribute_group, selected_features)))


# In[45]:


unique, counts = np.unique(attribute_types, return_counts=True)
print(np.asarray((unique, counts)).T)


# `TQWT` and `MFCC` features presented the most in selected features. They are also in top of feature weights:

# In[65]:


feature_weights = pd.DataFrame.from_records(list(zip(selected_features, *gs_logreg_fs.best_estimator_.named_steps.clf.coef_)), columns=['Feature', 'Weight'])


# In[84]:


feature_weights['Weight'] = feature_weights['Weight'].apply(np.abs)
feature_weights['Atribute group'] = feature_weights['Feature'].apply(check_attribute_group)


# Top 50 features by weight:

# In[85]:


feature_weights.sort_values(by='Weight', ascending=False, ignore_index=True).iloc[:50]


# # Results

# What has been done in this work:
# - EDA by attribute groups
# - Scaling method selection (StandardScaler was used)
# - Feature selection methods comparison (RFE with 150 features was used)
# 
# The final model scores:
# - Accuracy: 0.862745 
# - Recall: 0.964912
# - Precision: 0.866142
# - F1: 0.912863
# 
# Confusion matrix: 
# 
# ![image.png](attachment:1be04772-dd8c-4a9d-a11f-7bd2e2e61780.png)
