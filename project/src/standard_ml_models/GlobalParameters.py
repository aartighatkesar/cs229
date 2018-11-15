import sys
import pandas as pd
import numpy as np

class GlobalVariables:
    def __init__(self):
        if sys.platform=='win32':
            self.DATA_DIR="C:\\Users\\tihor\\Documents\\kaggle\\quora\\"
            self.OUTPUT_DIR='../submissions/'
            #self.DATA_DIR="d:\hk_futures\\"
            self.systemslash = "\\"
            self.train=pd.read_csv(self.DATA_DIR+'train.csv')
            self.test=pd.read_csv(self.DATA_DIR+'test.csv')
            # drop rows with nan
            self.train.dropna(inplace=True)
            self.test.replace(np.nan, '', regex=True, inplace=True)

def word_match_score(sentence1,sentence2,stopwords):
    sentence1Words={}
    sentence2Words={}
    for word in sentence1.lower().split():
        if word not in stopwords:
            sentence1Words[word]=1
    for word in sentence2.lower().split():
        if word not in stopwords:
            sentence2Words[word]=1
    if len(sentence1Words)==0 or len(sentence2Words)==0:
        return 0
    common_words_sentence1=[]
    common_words_sentence2=[]
    for word in sentence1Words:
        if word in sentence2Words:
            common_words_sentence1.append(word)
    for word in sentence2Words:
        if word in sentence1Words:
            common_words_sentence1.append(word)
    score=(len(common_words_sentence1)+len(common_words_sentence2))/(len(sentence1Words)+len(sentence2Words))
    return score

def tfidf_match(question1,question2,tfidfObject):
    if question1==np.nan or question2==np.nan:
        print(question1)
        print(question2)
        return 0
    vector1=tfidfObject.transform([question1]).toarray()
    vector2=tfidfObject.transform([question2]).toarray()
    score=(np.sum(vector1*vector2))/(np.sum(vector1)+np.sum(vector2))
    return score


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def tfidf_word_match_share(question1,question2,weights,stops):
    q1words = {}
    q2words = {}
    for word in question1.lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in question2.lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R