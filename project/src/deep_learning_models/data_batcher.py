from deep_learning_models.word_and_character_vectors import get_char,get_glove,PAD_ID,UNK_ID
from deep_learning_models.sentence_operations import sentence_to_word_ids,pad_words,convert_ids_to_word_vectors
from sklearn.model_selection import train_test_split
import random
import numpy as np
import GlobalParameters

class QuoraDataObject(object):
    def __init__(self,word2id,char2id,word_embed_matrix,char_embed_matrix,train_data,test_data,number_of_words_in_question,number_of_letters_in_word,batch_size,discard_long=False,test_size=0.1):
        temp_ids=test_data['test_id'].tolist()
        temp_q1=test_data['question1'].tolist()
        temp_q2=test_data['question2'].tolist()
        self.test_data=[]
        for i,item in enumerate(temp_ids):
            self.test_data.append((item,temp_q1[i],temp_q2[i]))
        self.word2id=word2id
        self.char2id=char2id
        self.word_embed_matrix=word_embed_matrix
        self.char_embed_matrix=char_embed_matrix
        self.batch_size=batch_size
        temp_train_data,temp_dev_data=train_test_split(train_data,test_size=test_size)
        temp_duplicates=temp_train_data['is_duplicate'].tolist()
        temp_q1=temp_train_data['question1'].tolist()
        temp_q2=temp_train_data['question2'].tolist()
        self.train_data=[]
        for i, item in enumerate(temp_duplicates):
            self.train_data.append((item, temp_q1[i], temp_q2[i]))
        temp_duplicates=temp_dev_data['is_duplicate'].tolist()
        temp_q1=temp_dev_data['question1'].tolist()
        temp_q2=temp_dev_data['question2'].tolist()
        self.dev_data=[]
        for i, item in enumerate(temp_duplicates):
            self.dev_data.append((item, temp_q1[i], temp_q2[i]))
        self.number_of_words_in_question=number_of_words_in_question
        self.number_of_letters_in_word=number_of_letters_in_word
        self.discard_long=discard_long

    def process_dataset(self,dataset):
        dataset.to_csv('temp.csv')
        dataset['q1_ids_padded']=dataset['question1'].apply(lambda x:pad_words(sentence_to_word_ids(x,self.word2id)[1],self.number_of_words_in_question))
        dataset['q2_ids_padded']=dataset['question2'].apply(lambda x:pad_words(sentence_to_word_ids(x,self.word2id)[1],self.number_of_words_in_question))
        dataset['q1_glove']=dataset['q1_ids_padded'].apply(lambda x:convert_ids_to_word_vectors(x,self.word_embed_matrix))
        dataset['q2_glove']=dataset['q2_ids_padded'].apply(lambda x:convert_ids_to_word_vectors(x,self.word_embed_matrix))


        q1_ids_padded = np.array(dataset['q1_ids_padded'].tolist())
        q2_ids_padded = np.array(dataset['q2_ids_padded'].tolist())
        q1_glove = np.array(dataset['q1_glove'].tolist())
        q2_glove = np.array(dataset['q2_glove'].tolist())
        q1_mask = (q1_ids_padded != PAD_ID).astype(np.int32)
        q2_mask = (q2_ids_padded != PAD_ID).astype(np.int32)
        return q1_ids_padded,q1_glove,q1_mask,q2_ids_padded,q2_glove,q2_mask

    def generate_one_epoch(self):
        num_batches=int(len(self.train_data))//self.batch_size
        if self.batch_size*num_batches<len(self.train_data): num_batches += 1
        random.shuffle(self.train_data)
        for i in range(num_batches):
            q1_words_for_mask=[]
            q1_words_vectorized=[]
            q2_words_for_mask=[]
            q2_words_vectorized=[]
            is_duplicates=[]
            #data_subset=self.train_data[i*self.batch_size:(i+1)*self.batch_size].copy()
            for is_duplicate,q1,q2 in self.train_data[i*self.batch_size:(i+1)*self.batch_size]:
                q1_tokens,q1_word_ids=sentence_to_word_ids(q1,self.word2id)
                q1_word_ids=pad_words(q1_word_ids,self.number_of_words_in_question)
                if len(q1_word_ids)!=self.number_of_words_in_question:
                    print("Incorrect length!!!!!")
                q1_word_ids_to_vectors=convert_ids_to_word_vectors(q1_word_ids,self.word_embed_matrix)
                q1_words_vectorized.append(q1_word_ids_to_vectors)
                q1_words_for_mask.append(q1_word_ids)

                q2_tokens,q2_word_ids=sentence_to_word_ids(q2, self.word2id)
                q2_word_ids=pad_words(q2_word_ids,self.number_of_words_in_question)
                if len(q2_word_ids) != self.number_of_words_in_question:
                    print("Incorrect length!!!!!")
                q2_word_ids_to_vectors=convert_ids_to_word_vectors(q2_word_ids,self.word_embed_matrix)
                q2_words_vectorized.append(q2_word_ids_to_vectors)
                q2_words_for_mask.append(q2_word_ids)
                is_duplicates.append(is_duplicate)
            q1_words_vectorized=np.array(q1_words_vectorized)
            q1_words_for_mask=np.array(q1_words_for_mask)
            q1_mask=(q1_words_for_mask != PAD_ID).astype(np.int32)
            q2_words_vectorized=np.array(q2_words_vectorized)
            q2_words_for_mask=np.array(q2_words_for_mask)
            q2_mask=(q2_words_for_mask != PAD_ID).astype(np.int32)
            is_duplicates=np.array(is_duplicates)
            is_duplicates = is_duplicates.reshape(-1, 1)
            yield q1_words_vectorized,q1_mask,q2_words_vectorized,q2_mask,is_duplicates

            #q1_ids_padded, q1_glove, q1_mask, q2_ids_padded, q2_glove, q2_mask=self.process_dataset(data_subset)
            #duplicate=np.array(data_subset['is_duplicate'].tolist())
            #duplicate=duplicate.reshape((-1,1))
            #yield q1_ids_padded,q1_glove,q1_mask,q2_ids_padded,q2_glove,q2_mask,duplicate

    def generate_dev_data(self):
        num_batches = int(len(self.dev_data)) // self.batch_size
        if self.batch_size * num_batches < len(self.dev_data): num_batches += 1
        for i in range(num_batches):
            q1_words_for_mask = []
            q1_words_vectorized = []
            q2_words_for_mask = []
            q2_words_vectorized = []
            is_duplicates = []
            for is_duplicate, q1, q2 in self.dev_data[i * self.batch_size:(i + 1) * self.batch_size]:
                q1_tokens, q1_word_ids = sentence_to_word_ids(q1, self.word2id)
                q1_word_ids = pad_words(q1_word_ids, self.number_of_words_in_question)
                if len(q1_word_ids) != self.number_of_words_in_question:
                    print("Incorrect length!!!!!")
                q1_word_ids_to_vectors = convert_ids_to_word_vectors(q1_word_ids, self.word_embed_matrix)
                q1_words_vectorized.append(q1_word_ids_to_vectors)
                q1_words_for_mask.append(q1_word_ids)

                q2_tokens, q2_word_ids = sentence_to_word_ids(q2, self.word2id)
                q2_word_ids = pad_words(q2_word_ids, self.number_of_words_in_question)
                if len(q2_word_ids) != self.number_of_words_in_question:
                    print("Incorrect length!!!!!")
                q2_word_ids_to_vectors = convert_ids_to_word_vectors(q2_word_ids, self.word_embed_matrix)
                q2_words_vectorized.append(q2_word_ids_to_vectors)
                q2_words_for_mask.append(q2_word_ids)
                is_duplicates.append(is_duplicate)
            q1_words_vectorized = np.array(q1_words_vectorized)
            q1_words_for_mask = np.array(q1_words_for_mask)
            q1_mask = (q1_words_for_mask != PAD_ID).astype(np.int32)
            q2_words_vectorized = np.array(q2_words_vectorized)
            q2_words_for_mask = np.array(q2_words_for_mask)
            q2_mask = (q2_words_for_mask != PAD_ID).astype(np.int32)
            is_duplicates = np.array(is_duplicates)
            is_duplicates=is_duplicates.reshape(-1,1)
            yield q1_words_vectorized, q1_mask, q2_words_vectorized, q2_mask, is_duplicates

    def generate_test_data(self):
        num_batches = int(len(self.test_data)) // self.batch_size
        if self.batch_size * num_batches < len(self.test_data): num_batches += 1
        for i in range(num_batches):
            q1_words_for_mask = []
            q1_words_vectorized = []
            q2_words_for_mask = []
            q2_words_vectorized = []
            test_ids = []
            for test_id, q1, q2 in self.test_data[i * self.batch_size:(i + 1) * self.batch_size]:
                q1_tokens, q1_word_ids = sentence_to_word_ids(q1, self.word2id)
                q1_word_ids = pad_words(q1_word_ids, self.number_of_words_in_question)
                if len(q1_word_ids) != self.number_of_words_in_question:
                    print("Incorrect length!!!!!")
                q1_word_ids_to_vectors = convert_ids_to_word_vectors(q1_word_ids, self.word_embed_matrix)
                q1_words_vectorized.append(q1_word_ids_to_vectors)
                q1_words_for_mask.append(q1_word_ids)

                q2_tokens, q2_word_ids = sentence_to_word_ids(q2, self.word2id)
                q2_word_ids = pad_words(q2_word_ids, self.number_of_words_in_question)
                if len(q2_word_ids) != self.number_of_words_in_question:
                    print("Incorrect length!!!!!")
                q2_word_ids_to_vectors = convert_ids_to_word_vectors(q2_word_ids, self.word_embed_matrix)
                q2_words_vectorized.append(q2_word_ids_to_vectors)
                q2_words_for_mask.append(q2_word_ids)
                test_ids.append(test_id)
            q1_words_vectorized = np.array(q1_words_vectorized)
            q1_words_for_mask = np.array(q1_words_for_mask)
            q1_mask = (q1_words_for_mask != PAD_ID).astype(np.int32)
            q2_words_vectorized = np.array(q2_words_vectorized)
            q2_words_for_mask = np.array(q2_words_for_mask)
            q2_mask = (q2_words_for_mask != PAD_ID).astype(np.int32)
            test_ids = np.array(test_ids )
            yield q1_words_vectorized, q1_mask, q2_words_vectorized, q2_mask, test_ids

#vars=GlobalParameters.GlobalVariables()
#emb_matrix_char, char2id, id2char=get_char('C:\\Users\\tihor\\Documents\\ml_data_files')
#emb_matrix_word, word2id, id2word=get_glove('C:\\Users\\tihor\\Documents\\ml_data_files')
#zz=QuoraDataObject(word2id=word2id,char2id=char2id,word_embed_matrix=emb_matrix_word,char_embed_matrix=emb_matrix_char,train_data=vars.train,test_data=vars.test,
#               number_of_words_in_question=238,number_of_letters_in_word=1176,batch_size=5000,test_size=0.1)
#for q1_words_vectorized,q1_mask,q2_words_vectorized,q2_mask,is_duplicates in zz.generate_test_data():
#    print(q1_words_vectorized.shape,q1_mask.shape,q2_words_vectorized.shape,q2_mask.shape,is_duplicates.shape)
#    break

#for review_words,review_chars,review_mask in zz.generate_test_data():
#    print(review_words.shape)
#    print(review_chars.shape)
#    print(review_mask.shape)
#    break