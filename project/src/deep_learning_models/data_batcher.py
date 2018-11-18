from deep_learning_models.word_and_character_vectors import PAD_ID,UNK_ID
from deep_learning_models.sentence_operations import get_ids_and_vectors
from sklearn.model_selection import train_test_split
import numpy as np

class QuoraObject(object):
    def __init__(self,word2id,char2id,word_embed_matrix,char_embed_matrix,train_data,test_data,number_of_words_in_question,number_of_letters_in_word,batch_size,test_size=0.1):
        self.test_data=test_data
        self.word2id=word2id
        self.char2id=char2id
        self.word_embed_matrix=word_embed_matrix
        self.char_embed_matrix=char_embed_matrix
        self.batch_size=batch_size
        self.train_data,self.dev_data=train_test_split(train_data,test_size=test_size)
        self.number_of_words_in_question=number_of_words_in_question
        self.number_of_letters_in_word=number_of_letters_in_word

    def generate_one_epoch(self):
        num_batches=int(len(self.train_data))//self.batch_size
        np.random.shuffle(self.train_data)
        for i in range(num_batches):
            data_subset=self.train_data[i*self.batch_size:(i+1)*self.batch_size]
