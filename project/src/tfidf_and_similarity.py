import pandas as pd
import GlobalParameters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from collections import Counter

#stops=set(ENGLISH_STOP_WORDS)
stops=set(stopwords.words("english"))

vars=GlobalParameters.GlobalVariables()

#all_qs=pd.Series(vars.train['question1'].tolist()+vars.train['question2'].tolist()+vars.test['question1'].tolist()+vars.test['question2'].tolist()).astype("str")
all_qs=pd.Series(vars.train['question1'].tolist()+vars.train['question2'].tolist()).astype("str")
tfidf=TfidfVectorizer(ngram_range=(1,1),stop_words=stops)
print("Fitting tfidf...")
tfidf.fit(all_qs)
print("Fitting tfidf...Done")

eps = 5000
words = (" ".join(all_qs)).lower().split()
counts = Counter(words)
weights = {word: GlobalParameters.get_weight(count) for word, count in counts.items()}

print("Setting wordmatch score...")
vars.train['wordmatch_score']=vars.train.apply(lambda row: GlobalParameters.word_match_score(row['question1'], row['question2'], stops), axis=1)
print("Setting wordmatch score...Done")
print("Setting tfidf score...")
vars.train['tfidf_score']=vars.train.apply(lambda row: GlobalParameters.tfidf_match(row['question1'], row['question2'], tfidf), axis=1)
print("Setting tfidf score...Done")
print("setting word_match_share...")
vars.train['anokas_score']=vars.train.apply(lambda row: GlobalParameters.tfidf_word_match_share(row['question1'], row['question2'],weights,stops), axis=1)
print("setting word_match_share...Done")
vars.train['length1']=vars.train['question1'].apply(lambda x:len(x))
vars.train['length2']=vars.train['question2'].apply(lambda x:len(x))
vars.train['numWords1']=vars.train['question1'].apply(lambda x:len(x.split()))
vars.train['numWords2']=vars.train['question2'].apply(lambda x:len(x.split()))

x_train=vars.train[['wordmatch_score','tfidf_score','anokas_score','length1','length2','numWords1','numWords2']]
y_train = vars.train['is_duplicate'].values
from sklearn.cross_validation import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

print("Setting test wordmatch score...")
vars.test['wordmatch_score']=vars.test.apply(lambda row: GlobalParameters.word_match_score(row['question1'], row['question2'], stops), axis=1)
print("Setting test wordmatch score...Done")
print("Setting test tfidf score...")
vars.test['tfidf_score']=vars.test.apply(lambda row: GlobalParameters.tfidf_match(row['question1'], row['question2'], tfidf), axis=1)
print("Setting test tfidf score...Done")
print("setting test word_match_share...")
vars.test['anokas_score']=vars.test.apply(lambda row: GlobalParameters.tfidf_word_match_share(row['question1'], row['question2'],weights,stops), axis=1)
print("setting test word_match_share...Done")
vars.test['length1']=vars.test['question1'].apply(lambda x:len(x))
vars.test['length2']=vars.test['question2'].apply(lambda x:len(x))
vars.test['numWords1']=vars.test['question1'].apply(lambda x:len(x.split()))
vars.test['numWords2']=vars.test['question2'].apply(lambda x:len(x.split()))

x_test=vars.test[['wordmatch_score','tfidf_score','anokas_score','length1','length2','numWords1','numWords2']]

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = vars.test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('submissions/tfidf_and_similarity_with_xgb.csv', index=False)