from deep_learning_models.word_and_character_vectors import get_glove,get_char
import tensorflow as tf
import os
from deep_learning_models.quora_model import QuoraVanillaNeuralNetworkModel

ML_DATA_FILES="c:\\Users\\tihor\\Documents\\ml_data_files\\"

tf.app.flags.DEFINE_integer("gpu", 1, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_integer("num_epochs",100, "Number of epochs to train. 0 means train indefinitely")

tf.app.flags.DEFINE_float("learning_rate",0.001,"Learning rate.")
tf.app.flags.DEFINE_float("dropout",0.5,"Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size",8000,"Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size",200,"Size of the hidden states")
tf.app.flags.DEFINE_integer("number_of_words_in_question",237,"The maximum words in each question")
tf.app.flags.DEFINE_integer("number_of_letters_in_word", 1176, "The maximum characters in each word")
tf.app.flags.DEFINE_integer("word_embedding_size", 300, "Size of the pretrained word vectors.")
#tf.app.flags.DEFINE_integer("char_embedding_size", 128, "Size of the pretrained char vectors.")
tf.app.flags.DEFINE_float("test_size",0.10,"Dev set to split from training set")
tf.app.flags.DEFINE_bool("discard_long",False,"Discard lines longer than review_length")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

glove_word_embed_matrix,glove_word2id,glove_id2word=get_glove(ML_DATA_FILES)
char_embed_matrix,char_char2id,char_id2char=get_char(ML_DATA_FILES)

qm=QuoraVanillaNeuralNetworkModel(FLAGS,glove_word2id,glove_word_embed_matrix)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for epoch in range(FLAGS.num_epochs):
    validation_accuracy=qm.run_epoch(sess)
    print('validation_accuracy for epoch ' + str(epoch) + ' => ' + str(validation_accuracy))
print('Final validation_accuracy => ' +str(qm.get_validation_accuracy(sess)))