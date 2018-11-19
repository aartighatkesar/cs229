from deep_learning_models.modules import NeuralNetworkHiddenLayer
from deep_learning_models.model import Model
from deep_learning_models.data_batcher import QuoraDataObject
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import GlobalParameters

vars=GlobalParameters.GlobalVariables()

class QuoraVanillaNeuralNetworkModel(Model):
    def __init__(self,FLAGS,word2id,word_embed_matrix):
        self.FLAGS=FLAGS
        self.dataObject=QuoraDataObject(word2id=word2id,char2id=None,word_embed_matrix=word_embed_matrix,char_embed_matrix=None,train_data=vars.train,test_data=vars.test,
               number_of_words_in_question=FLAGS.number_of_words_in_question,number_of_letters_in_word=FLAGS.number_of_letters_in_word,batch_size=FLAGS.batch_size,
                                        discard_long=FLAGS.discard_long,test_size=FLAGS.test_size)
        with tf.variable_scope("QuoraModel",initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,uniform=True)):
            self.add_placeholders()
            self.build_graph()
            self.add_loss()
            self.add_training_step()

    def add_placeholders(self):
        self.q1_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.number_of_words_in_question,self.FLAGS.word_embedding_size])
        self.q1_mask=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.number_of_words_in_question])
        self.q2_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.number_of_words_in_question,self.FLAGS.word_embedding_size])
        self.q2_mask=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.number_of_words_in_question])
        self.is_duplicate=tf.placeholder(dtype=tf.float32,shape=[None,1])
        self.keep_prob=tf.placeholder_with_default(1.0,shape=())

    def build_graph(self):
        HIDDEN_LAYER1_SIZE=2048
        HIDDEN_LAYER2_SIZE=2048
        HIDDEN_LAYER3_SIZE=2048

        q1_words_sum=tf.reduce_sum(self.q1_words,1)   #(batch,word_embedding_size)
        q1_mask_sum=tf.reduce_sum(self.q1_mask,1,keep_dims=True)+tf.constant(0.0001,dtype=tf.float32,shape=())  #(batch,1)
        input_vector1=q1_words_sum/q1_mask_sum  #(batch,word_embedding_size)

        q2_words_sum=tf.reduce_sum(self.q2_words, 1)  # (batch,word_embedding_size)
        q2_mask_sum=tf.reduce_sum(self.q2_mask, 1, keep_dims=True) + tf.constant(0.0001, dtype=tf.float32,shape=())  # (batch,1)
        input_vector2 = q2_words_sum/q2_mask_sum

        q1_layer1=NeuralNetworkHiddenLayer('Q1_HiddenLayer1', self.FLAGS.word_embedding_size, HIDDEN_LAYER1_SIZE,self.keep_prob)
        q1_output1=q1_layer1.build_graph(input_vector1)  # (batch,hidden_layer1_size)

        q1_layer2=NeuralNetworkHiddenLayer('Q1_HiddenLayer2', HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE, self.keep_prob)
        q1_output2=q1_layer2.build_graph(q1_output1)  # (batch,hidden_layer2_size)

        q1_layer3=NeuralNetworkHiddenLayer('Q1_HiddenLayer3', HIDDEN_LAYER2_SIZE, HIDDEN_LAYER3_SIZE, self.keep_prob)
        q1_output3=q1_layer3.build_graph(q1_output2)  # (batch,hidden_layer3_size)

        q2_layer1=NeuralNetworkHiddenLayer('Q2_HiddenLayer1',self.FLAGS.word_embedding_size,HIDDEN_LAYER1_SIZE,self.keep_prob)
        q2_output1=q2_layer1.build_graph(input_vector2)  # (batch,hidden_layer1_size)

        q2_layer2=NeuralNetworkHiddenLayer('Q2_HiddenLayer2', HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE, self.keep_prob)
        q2_output2=q2_layer2.build_graph(q2_output1)  # (batch,hidden_layer2_size)

        q2_layer3=NeuralNetworkHiddenLayer('Q2_HiddenLayer3', HIDDEN_LAYER2_SIZE, HIDDEN_LAYER3_SIZE, self.keep_prob)
        q2_output3=q2_layer3.build_graph(q2_output2)  # (batch,hidden_layer3_size)

        #concatenate the 2 networks
        blended_reps=tf.concat([q1_output3,q2_output3],axis=1)  # (batch_size, hidden_layer3_size*2)
        with tf.variable_scope('full_layer1') as scope:
            full_weight1=tf.get_variable(name='full_layer1_weight',shape=[HIDDEN_LAYER3_SIZE*2,1], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05))
            full_bias1=tf.get_variable(name='full_layer_bias', shape=[1], dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            final_output = tf.matmul(blended_reps, full_weight1) + full_bias1
        logits = tf.identity(final_output, name="logits")
        self.final_output = final_output
        self.logits = logits

    def add_loss(self):
        with vs.variable_scope('loss'):
            #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.ratings))
            self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.is_duplicate))
            self.correct_prediction = tf.equal(tf.round(self.final_output), self.is_duplicate,name='correct_prediction')
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')
            tf.summary.scalar('cost', self.cost)

    def add_training_step(self):
        self.train_step=tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate).minimize(self.cost)

    def run_train_iter(self,sess,input_batch,is_duplicate_batch):
        q1_words_vectorized_batch,q1_mask_batch,q2_words_vectorized_batch,q2_mask_batch=input_batch
        train_data_feed={
            self.q1_words:q1_words_vectorized_batch,
            self.q1_mask:q1_mask_batch,
            self.q2_words:q2_words_vectorized_batch,
            self.q2_mask:q2_mask_batch,
            self.is_duplicate:is_duplicate_batch,
            self.keep_prob:(1.0-self.FLAGS.dropout),
        }
        sess.run(self.train_step,feed_dict=train_data_feed)

    def get_validation_accuracy(self,sess):
        validation_accuracy=0.0
        num_batches=0.0
        for q1_words_vectorized_batch,q1_mask_batch,q2_words_vectorized_batch,q2_mask_batch, is_duplicate_batch in self.dataObject.generate_dev_data():
            num_batches+=1
            dev_data_feed={
                self.q1_words: q1_words_vectorized_batch,
                self.q1_mask: q1_mask_batch,
                self.q2_words: q2_words_vectorized_batch,
                self.q2_mask: q2_mask_batch,
                self.is_duplicate: is_duplicate_batch,
                self.keep_prob: 1.0,
            }
            validation_accuracy_batch=sess.run([self.accuracy],dev_data_feed)
            validation_accuracy+=validation_accuracy_batch[0]
        validation_accuracy/=num_batches
        return validation_accuracy

    def get_test_data(self,sess):
        output=[]
        lineids=[]
        for q1_words_vectorized_batch,q1_mask_batch,q2_words_vectorized_batch,q2_mask_batch, line_ids_batch in self.dataObject.generate_test_data():
            test_data_feed={
                self.q1_words: q1_words_vectorized_batch,
                self.q1_mask: q1_mask_batch,
                self.q2_words: q2_words_vectorized_batch,
                self.q2_mask: q2_mask_batch,
                self.keep_prob: 1.0,
            }
            test_output=sess.run(self.final_output,feed_dict=test_data_feed)
            lineids.extend(line_ids_batch.tolist())
            output.extend(test_output.tolist())
        return lineids,output

    def run_epoch(self,sess):
        for q1_words_vectorized,q1_mask,q2_words_vectorized,q2_mask,is_duplicates in self.dataObject.generate_one_epoch():
            input_batch=(q1_words_vectorized,q1_mask,q2_words_vectorized,q2_mask)
            self.run_train_iter(sess,input_batch,is_duplicates)
        validation_accuracy=self.get_validation_accuracy(sess)
        return validation_accuracy