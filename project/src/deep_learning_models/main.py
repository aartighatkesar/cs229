from deep_learning_models.word_and_character_vectors import get_glove,get_char
import GlobalParameters
from deep_learning_models.data_batcher import QuoraObject

ML_DATA_FILES="c:\\Users\\tihor\\Documents\\ml_data_files\\"

#vars=GlobalParameters.GlobalVariables

#glove_word_embed_matrix,glove_word2id,glove_id2word=get_glove(ML_DATA_FILES)
char_embed_matrix,char_char2id,char_id2char=get_char(ML_DATA_FILES)

#data=QuoraObject(glove_word2id,char_char2id,glove_word_embed_matrix,char_embed_matrix,vars.train_data,vars.test_data,238,149,100,test_size=0.1)
from deep_learning_models.sentence_operations import sentence_to_char_ids
text='this is a test of conversion'
tokens,char_ids=sentence_to_char_ids(text,char_char2id)