'''

The code was submitted to kaggle using the lgbm native algorithm to experiment if it has
 better performance results than the sklearn library. 
 Its kaggle score is: 0.7346,but the  result is disappointing
'''


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import re
import string
from wordcloud import STOPWORDS
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import FreqDist, word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer


path1='F:\kaggle\kaggle_nlp\\train.csv'
path2='F:\kaggle\kaggle_nlp\\test.csv'
path3='F:\kaggle\kaggle_nlp\\sample_submission.csv'
# data = pd.read_csv(path)
# # data=data.remove('datetime')
# print(data.head())
plt.rcParams.update({'font.size': 12})


df_train = pd.read_csv(path1)
df_test = pd.read_csv(path2)
sub_sample = pd.read_csv(path3)

print (df_train.shape, df_test.shape, sub_sample.shape)

# Fill missing values with 'None'
df_train['keyword'] = df_train['keyword'].fillna(f'None')
df_test['keyword'] = df_test['keyword'].fillna(f'None')

# fix '20%' typo in 'keyword' column
df_train['keyword'] = df_train['keyword'].apply(lambda x: re.sub('%20', ' ', x))
df_test['keyword'] = df_test['keyword'].apply(lambda x: re.sub('%20', ' ', x))

no_keyword = df_train['keyword'] == 'None'
keywords = np.unique(df_train[~no_keyword]['keyword'].to_numpy())

for df in [df_train, df_test]:
    for i in range(len(df)):
        if df.loc[i, 'keyword'] == 'None':
            for k in keywords:
                if k in df.loc[i, 'text'].lower():
                    df.loc[i, 'keyword'] = k
                    break


print('Number of missing values left:')
print('For training set:', df_train[df_train['keyword'] == 'None'].shape[0])
print('For test set:', df_test[df_test['keyword'] == 'None'].shape[0])

pd.concat([df_train[df_train['keyword'] == 'None']['text'], df_test[df_test['keyword'] == 'None']['text']])




# Fill missing values with 'None'
df_train['location'] = df_train['location'].fillna(f'None')
df_test['location'] = df_test['location'].fillna(f'None')


stop_words = set(list(STOPWORDS) + stopwords.words('english'))

def clean_text(text):
    # remove unicode character (Â‰Ã, \x89Û_, ...)
    text = text.encode("ascii", "ignore")
    text = text.decode()

    # remove old style retweet text "RT" and "#RT"
    text = re.sub(r'#RT\s*', '', text) # "#RT"
    text = text.replace("RT ", "", 1) # "RT"

    text = text.lower() # lowercase
    text = re.sub(r'\n',' ', text) # remove line breaks
    text = re.sub(r'https?://\S+', '', text) # remove link
    text = re.sub(r'@\S+', '', text) # remove mention
    text = re.sub('[0-9]+', '', text) # remove number
    text = re.sub('\s*&\S+\s*', ' ', text) # remove ampersand codes (&lt;b&gt;&lt;i&gt;&lt;u&gt;&amp)

    # remove stopwords
    filtered_text = [w for w in text.split() if not w in stop_words]
    text = ' '.join(filtered_text)

    text = re.sub(r'[^\w\s]', '', text) # remove punctuations
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces

    return text


# train set
df_train['text_clean'] = df_train['text'].apply(lambda x: clean_text(x))

# test set
df_test['text_clean'] = df_test['text'].apply(lambda x: clean_text(x))

print("df_train['text_clean_1']:",df_train['text_clean'].head())


# word_count
df_train['word_count'] = df_train['text'].apply(lambda x: len(str(x).split()))
df_test['word_count'] = df_test['text'].apply(lambda x: len(str(x).split()))

# unique_word_count
df_train['unique_word_count'] = df_train['text'].apply(lambda x: len(set(str(x).split())))
df_test['unique_word_count'] = df_test['text'].apply(lambda x: len(set(str(x).split())))

# stop_word_count
df_train['stop_word_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
df_test['stop_word_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

# url_count
df_train['url_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
df_test['url_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

# mean_word_length
df_train['mean_word_length'] = df_train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df_test['mean_word_length'] = df_test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# char_count
df_train['char_count'] = df_train['text'].apply(lambda x: len(str(x)))
df_test['char_count'] = df_test['text'].apply(lambda x: len(str(x)))

# punctuation_count
df_train['punctuation_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df_test['punctuation_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# hashtag_count
df_train['hashtag_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
df_test['hashtag_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

# mention_count
df_train['mention_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
df_test['mention_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

# # Vectorize with TfidfVectorizer (only include >=5 occurrences; unigrams and bigrams)
# vec_text = TfidfVectorizer(min_df = 5, ngram_range = (1,2), stop_words='english')
#
# text_vec_train = vec_text.fit_transform(df_train['text_clean_1'])
# X_train = pd.DataFrame(text_vec_train.toarray(), columns=vec_text.get_feature_names())
#
# text_vec_test = vec_text.transform(df_test['text_clean_1'])
# X_test = pd.DataFrame(text_vec_test.toarray(), columns=vec_text.get_feature_names())
# y_train = df_train['target']
# X_test.shape
# print('X_test.shape:',X_test.shape)

def clean_text_1(text):
    # remove unicode character (Â‰Ã, \x89Û_, ...)
    text = text.encode("ascii", "ignore")
    text = text.decode()

    # remove old style retweet text "RT" and "#RT"
    text = re.sub(r'#RT\s*', '', text) # "#RT"
    text = text.replace("RT ", "", 1) # "RT"

    text = text.lower() # lowercase
    text = re.sub(r'\n',' ', text) # remove line breaks
    text = re.sub(r'https?://\S+', '', text) # remove link
#     text = re.sub(r'@\S+', '', text) # remove mention
    text = re.sub('[0-9]+', '', text) # remove number
    text = re.sub('\s*&\S+\s*', ' ', text) # remove ampersand codes (&lt;b&gt;&lt;i&gt;&lt;u&gt;&amp)
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces

    text = re.sub(' go ', ' ', text)
    text = re.sub(' went ', ' ', text)
    text = re.sub(' goes ', ' ', text)
    text = re.sub(' people ', ' ', text)

    return text


# train set
df_train['text_clean_1'] = df_train['text'].apply(lambda x: clean_text_1(x))

# test set
df_test['text_clean_1'] = df_test['text'].apply(lambda x: clean_text_1(x))

print("df_train['text_clean_1']:",df_train['text_clean_1'])

print("df_test['text_clean_1']:",df_test['text_clean_1'])


# Vectorize with TfidfVectorizer (only include >=5 occurrences; unigrams and bigrams)
vec_text = TfidfVectorizer(min_df = 5, ngram_range = (1,2), stop_words='english')

text_vec_train = vec_text.fit_transform(df_train['text_clean_1'])
X_train = pd.DataFrame(text_vec_train.toarray(), columns=vec_text.get_feature_names())
print('X_train.head():',X_train.head(50))
print('X_train.columns.tolist()=',X_train.columns.tolist())

text_vec_test = vec_text.transform(df_test['text_clean_1'])
X_test = pd.DataFrame(text_vec_test.toarray(), columns=vec_text.get_feature_names())
y_train = df_train['target']
print('X_test.shape:',X_test.shape)



scaler = MinMaxScaler()
# scaler = StandardScaler()

# Scale text first
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Scale mention_count
scaler.fit(df_train['mention_count'].to_numpy().reshape(-1, 1))
mention_count_train = scaler.transform(df_train['mention_count'].to_numpy().reshape(-1, 1))
mention_count_test = scaler.transform(df_test['mention_count'].to_numpy().reshape(-1, 1))
X_train = np.concatenate((X_train, mention_count_train), axis=1)
X_test = np.concatenate((X_test, mention_count_test), axis=1)
print('=========')

# 转出dataframe
X_train=pd.DataFrame(X_train)
y_train=pd.DataFrame(y_train)
X_test=pd.DataFrame(X_test)
print(type(X_train))
print(X_train.head())
print(y_train)
print('aaaaaaaaaaaaaaaaaaaaaaaa')
features = [i for i in X_train.columns]
# 增加一列
X_train['target']=y_train
# X_train=X_train.loc[:,'lable']=y_train.values
# X_train=pd.concat([X_train, y_train], sort=False)
# X_train = pd.DataFrame( data=[[X_train,y_train]])
print(X_train)
print(X_train.shape)
print('bbbbbbbbbbbbbbbb')
train=X_train
# 准备

train_x= X_train[features]
train_y = X_train['target'].values
test = X_test[features]
print(train_x.shape)
print(train_y.shape)
print(test.shape)
print(test.head(2))


from sklearn.ensemble import AdaBoostClassifier
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import KFold
import lightgbm as lgb
# from lightgbm import LGBMClassifier



params = {'num_leaves': 60, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
          'min_data_in_leaf': 30,
          'objective': 'binary', #定义的目标函数
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,  #提取的特征比率
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,             #l1正则
          # 'lambda_l2': 0.001,     #l2正则
          "verbosity": -1,
          "nthread": -1,                #线程数量，-1表示全部线程，线程越多，运行的速度越快
          'metric': {'binary_logloss', 'auc'},  ##评价函数选择
          "random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
          # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
          }

folds = KFold(n_splits=5, shuffle=True, random_state=2019)
prob_oof = np.zeros((X_train.shape[0], ))
test_pred_prob = np.zeros((X_test.shape[0], ))


params = {'num_leaves': 60, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
          'min_data_in_leaf': 30,
          'objective': 'binary', #定义的目标函数
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,  #提取的特征比率
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,             #l1正则
          # 'lambda_l2': 0.001,     #l2正则
          "verbosity": -1,
          "nthread": -1,                #线程数量，-1表示全部线程，线程越多，运行的速度越快
          'metric': {'binary_logloss', 'auc'},  ##评价函数选择
          "random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
          # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
          }

folds = KFold(n_splits=5, shuffle=True, random_state=2019)
prob_oof = np.zeros((train_x.shape[0], ))
test_pred_prob = np.zeros((test.shape[0], ))


# train and predict
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train)):
    print("fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y[trn_idx])
    val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y[val_idx])


    clf = lgb.train(params,
                    trn_data,
                    100000,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=20,
                    early_stopping_rounds=60)
    prob_oof[val_idx] = clf.predict(train_x.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    test_pred_prob += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

res_df=pd.DataFrame()
res_df['id']=df_test['id']


threshold = 0.5
pred_df = pd.DataFrame()
for pred in test_pred_prob:
    print(pred)
    pred = 1 if pred > threshold else 0
    print(pred)
    pred_df = pred_df.append(pd.DataFrame({'target': pred}, index=[0]), ignore_index=True)
    # res_df['target']result
    # res_df.to_csv('submission.csv',index=False)

print(pred_df)
res_df['target']=pred_df
print(res_df)
res_df.to_csv('submission.csv',index=False)  # kaggle score:0.7346