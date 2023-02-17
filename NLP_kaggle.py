'''
The codes are able to reproduce the results of the paper experiments 
kaggle competition link: https://www.kaggle.com/competitions/nlp-getting-started/overview
Dataset from: https://www.kaggle.com/competitions/nlp-getting-started/data

Using Support Vector Machine, Random Forest, Decision Tree, Logistic Regression, AdaBoost, GBDT, LightGBM
and other algorithms to do disaster prediction on tweeted data.

The code for feature engineering, data cleaning, and partial algorithm implementation is as follows.

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
print('X_test:',X_test)
y_train = df_train['target']
print('y_train=',y_train)
print('X_test.shape:',X_test.shape)
print('X_test.head():',X_test.head(50))

print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

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
print(type(X_train))
print(X_train)


# print('X_train.columns.tolist()=',X_train.columns)
# print('X_test.columns.tolist()=',X_test.columns)
print('=========')

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
from sklearn.ensemble import  GradientBoostingClassifier


X_train,X_valid,y_train,y_valid=train_test_split(X_train,y_train,random_state=66)
def plot():


    lr = LogisticRegression(random_state=2)
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    lgbm=LGBMClassifier()
    adab=AdaBoostClassifier()
    svc=SVC()
    gbdt = GradientBoostingClassifier()

    # fit model
    lr.fit(X_train, y_train)
    dtc.fit(X_train,y_train)
    rfc.fit(X_train,y_train)

    lgbm.fit(X_train,y_train)
    adab.fit(X_train,y_train)
    svc.fit(X_train,y_train)
    gbdt.fit(X_train,y_train)

    # score
    score_lr = lr.score(X_valid, y_valid)
    score_dtc=dtc.score(X_valid,y_valid)
    score_rfc=rfc.score(X_valid,y_valid)

    score_lgbm=lgbm.score(X_valid,y_valid)
    score_adab=adab.score(X_valid,y_valid)
    score_svc=svc.score(X_valid,y_valid)
    score_gbdt=gbdt.score(X_valid, y_valid)

    print("logic regression:{}".format(score_lr))
    print("Single Tree:{}".format(score_dtc))
    print("Random Forest:{}".format(score_rfc))

    print("lgbm classifier:{}".format(score_lgbm))
    print("adab classifier:{}".format(score_adab))
    print("svc classifier:{}".format(score_svc))

    print("gbdt classifier:{}".format(score_gbdt))

    print('============starting========')

    lr_s = cross_val_score(lr, X_train, y_train, cv=15)
    dtc_s = cross_val_score(dtc, X_train,y_train, cv=15)
    rfc_s = cross_val_score(rfc, X_train,y_train, cv=15)

    lgbm_s = cross_val_score(lgbm, X_train,y_train, cv=15)
    adab_s = cross_val_score(adab, X_train,y_train, cv=15)
    svc_s = cross_val_score(svc, X_train,y_train, cv=15)
    gbdt_s = cross_val_score(gbdt, X_train, y_train, cv=15)

    # rfc_s = cross_val_score(rfc, X_train, y_train, cv=15)
    # clf_s = cross_val_score(clf, X_train, y_train, cv=15)
    # lr_s = cross_val_score(lr, X_train, y_train, cv=15)
    # lgbm_s = cross_val_score(lgbm, X_train, y_train, cv=15)

    # plot
    plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(range(1, 16), lr_s, color='yellow', label="logic regression")
    plt.plot(range(1, 16), dtc_s, color='blue', label="DecisionTreeClassifier")
    plt.plot(range(1, 16), rfc_s, color='red', label="RandomForestClassifier")

    plt.plot(range(1, 16), lgbm_s, color='black', label="lgbm Classifier")
    plt.plot(range(1, 16), adab_s, color='orange', label="sdaboost Classifier")
    plt.plot(range(1, 16), svc_s, color='gray', label="svc ")
    plt.plot(range(1, 16), gbdt_s, color='purple', label="gbdt ")
    plt.xlabel('Number of cross-validations')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()



def algorithm():

    lr = LogisticRegression(random_state=2)
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    lgbm=LGBMClassifier()
    adab=AdaBoostClassifier()
    svc=SVC()
    gbdt=GradientBoostingClassifier(random_state=10)

    lr.fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    rfc.fit(X_train, y_train)
    lgbm.fit(X_train,y_train)
    adab.fit(X_train,y_train)
    svc.fit(X_train,y_train)
    gbdt.fit(X_train,y_train)


    y_test1 = lr.predict(X_test)
    y_test2 = dtc.predict(X_test)
    y_test3 = rfc.predict(X_test)
    y_test4 = lgbm.predict(X_test)
    y_test5 = adab.predict(X_test)
    y_test6 = svc.predict(X_test)
    y_test7 = gbdt.predict(X_test)

    print('Training accuracy lr: %.8f' % lr.score(X_train, y_train))
    print('Training accuracy dtc: %.8f' % dtc.score(X_train, y_train))
    print('Training accuracy: rfc:%.8f' % rfc.score(X_train, y_train))
    print('Training accuracy lgbm: %.8f' % lgbm.score(X_train, y_train))
    print('Training accuracy adab: %.8f' % adab.score(X_train, y_train))
    print('Training accuracy: svc%.8f' % svc.score(X_train, y_train))
    print('Training accuracy: gbdt%.8f' % gbdt.score(X_train, y_train))
    submit = sub_sample.copy()
    # submit6 = sub_sample.copy()
    submit7 = sub_sample.copy()
    submit.target = y_test1  # kaggle 分数 0.79528  lr: 0.87797189
    submit.target = y_test2  # kaggle 分数 0.72448  dtc: 0.98029686
    submit.target = y_test3  # kaggle 分数 0.78363  rfc:0.98029686
    submit.target = y_test4  # kaggle 分数 0.78608  lgbm: 0.84368843
    submit.target = y_test5  # kaggle 分数 0.73521  adab: 0.76119795
    submit6.target = y_test6 # kaggle 分数 0.79528  svc0.91383160
    submit7.target = y_test7  # kaggle 分数 0.72418  gbdt0.77104952
    submit.to_csv('submission.csv',index=False)
    submit6.to_csv('submission6.csv', index=False)
    submit7.to_csv('submission7.csv', index=False)




if __name__ == '__main__':
    # algorithm()
    plot()
    algorithm()