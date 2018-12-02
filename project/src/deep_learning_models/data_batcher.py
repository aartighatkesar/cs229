from deep_learning_models.word_and_character_vectors import PAD_ID,UNK_ID
from deep_learning_models.sentence_operations import sentence_to_word_ids,pad_words,convert_ids_to_word_vectors
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

def get_ids_and_pad(text,word2id,num_words,discard_long=False):
    tokens, word_ids = sentence_to_word_ids(text,word2id)
    if len(word_ids)>num_words:
        if discard_long:
            return None,None
        else:
            tokens=tokens[:num_words]
            word_ids=word_ids[:num_words]
    else:
        word_ids=pad_words(word_ids, num_words)
    return word_ids

class QuoraDataObject(object):
    def __init__(self,word2id,char2id,train_data,test_data,number_of_words_in_question,number_of_letters_in_word,batch_size,stopwords=set(),discard_long=False,test_size=0.1):
        self.word2id=word2id
        self.char2id=char2id
        self.batch_size=batch_size
        self.train,self.dev=train_test_split(train_data,test_size=test_size)
        self.test=test_data
        #temp_train_data=temp_train_data.copy()
        #temp_dev_data=temp_dev_data.copy()

        self.number_of_words_in_question = number_of_words_in_question
        self.number_of_letters_in_word = number_of_letters_in_word
        self.discard_long = discard_long
        #temp_train_data['q1_word_ids']=temp_train_data['question1'].apply(lambda x:get_ids_and_pad(x,self.word2id,self.number_of_words_in_question,self.discard_long))
        #temp_train_data['q2_word_ids']=temp_train_data['question2'].apply(lambda x: get_ids_and_pad(x, self.word2id, self.number_of_words_in_question, self.discard_long))
        #temp_dev_data['q1_word_ids'] = temp_dev_data['question1'].apply(lambda x: get_ids_and_pad(x, self.word2id, self.number_of_words_in_question, self.discard_long))
        #temp_dev_data['q2_word_ids'] = temp_dev_data['question2'].apply(lambda x: get_ids_and_pad(x, self.word2id, self.number_of_words_in_question, self.discard_long))
        #test_data['q1_word_ids'] = test_data['question1'].apply(lambda x: get_ids_and_pad(x, self.word2id, self.number_of_words_in_question, self.discard_long))
        #test_data['q2_word_ids'] = test_data['question2'].apply(lambda x: get_ids_and_pad(x, self.word2id, self.number_of_words_in_question, self.discard_long))
        #self.train=temp_train_data[['q1_word_ids','q2_word_ids','is_duplicate']].copy()
        #self.dev=temp_dev_data[['q1_word_ids','q2_word_ids','is_duplicate']].copy()
        #print(test_data.shape)
        #print(test_data.head())
        #print(test_data.columns)
        #self.test=test_data['q1_word_ids,q2_word_ids,test_id'].copy()

    def generate_one_epoch(self):
        num_batches=int(self.train.shape[0])//self.batch_size
        if self.batch_size*num_batches<self.train.shape[0]: num_batches += 1
        self.train=shuffle(self.train)
        for i in range(num_batches):
            train_subset=self.train[i*self.batch_size:(i+1)*self.batch_size]
            q1_ids=np.array(train_subset['q1_word_ids'].tolist())
            q2_ids=np.array(train_subset['q2_word_ids'].tolist())
            q1_mask = (q1_ids != PAD_ID).astype(np.int32)
            q2_mask = (q2_ids != PAD_ID).astype(np.int32)
            labels=np.array(train_subset['is_duplicate'].tolist())
            labels = labels.reshape((labels.shape[0], 1))

            yield q1_ids,q2_ids,q1_mask,q2_mask,labels

    def generate_dev_data(self):
        num_batches = int(self.dev.shape[0]) // self.batch_size
        if self.batch_size * num_batches < self.dev.shape[0]: num_batches += 1
        for i in range(num_batches):
            dev_subset = self.dev[i * self.batch_size:(i + 1) * self.batch_size]
            q1_ids = np.array(dev_subset['q1_word_ids'].tolist())
            q2_ids = np.array(dev_subset['q2_word_ids'].tolist())
            q1_mask = (q1_ids != PAD_ID).astype(np.int32)
            q2_mask = (q2_ids != PAD_ID).astype(np.int32)
            labels = np.array(dev_subset['is_duplicate'].tolist())
            labels = labels.reshape((labels.shape[0], 1))

            yield q1_ids, q2_ids, q1_mask, q2_mask, labels

    def generate_test_data(self):
        num_batches = int(self.test.shape[0]) // self.batch_size
        if self.batch_size * num_batches < self.test.shape[0]: num_batches += 1
        for i in range(num_batches):
            test_subset = self.test[i * self.batch_size:(i + 1) * self.batch_size]
            q1_ids = np.array(test_subset ['q1_word_ids'].tolist())
            q2_ids = np.array(test_subset ['q2_word_ids'].tolist())
            q1_mask = (q1_ids != PAD_ID).astype(np.int32)
            q2_mask = (q2_ids != PAD_ID).astype(np.int32)
            labels = np.array(test_subset ['test_id'].tolist())
            labels=labels.reshape((labels.shape[0],1))
            yield q1_ids, q2_ids, q1_mask, q2_mask, labels


#import GlobalParameters
#from deep_learning_models.word_and_character_vectors import get_glove
#vars=GlobalParameters.GlobalVariables()
##emb_matrix_char, char2id, id2char=get_char('C:\\Users\\tihor\\Documents\\ml_data_files')
#emb_matrix_word, word2id, id2word=get_glove('C:\\Users\\tihor\\Documents\\ml_data_files')
#zz=QuoraDataObject(word2id=word2id,char2id=None,train_data=vars.train,test_data=vars.test,number_of_words_in_question=30,number_of_letters_in_word=100,batch_size=5000,test_size=0.1)
#for q1_ids, q2_ids, q1_mask, q2_mask, labels in zz.generate_one_epoch():
#    print(q1_ids.shape,q2_ids.shape,q1_mask.shape,q2_mask.shape,labels.shape)
#    print(q1_ids, q2_ids, q1_mask, q2_mask, labels)
#    break

#for review_words,review_chars,review_mask in zz.generate_test_data():
#    print(review_words.shape)
#    print(review_chars.shape)
#    print(review_mask.shape)
#    break