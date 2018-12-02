from deep_learning_models.modules import BiRNNLayer
from deep_learning_models.model import Model
from deep_learning_models.data_batcher import QuoraDataObject
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops
#import GlobalParameters
import pandas as pd

def cosine_similarity(q1,q2):
    q1_norm=tf.math.l2_normalize(q1,axis=1)
    q2_norm = tf.math.l2_normalize(q2, axis=1)
    return tf.reduce_sum((q1_norm*q2_norm),axis=1,keepdims=True)

class QuoraBiRNNModel(Model):
    def __init__(self,FLAGS,id2word,word2id,emb_matrix):
        print("Initializing the model")
        self.FLAGS=FLAGS
        self.id2word=id2word
        self.word2id=word2id
        train=pd.read_pickle('../train.pkl')
        test=pd.read_pickle('../test.pkl')
        self.dataObject=QuoraDataObject(word2id=word2id,char2id=None,train_data=train,test_data=test,number_of_words_in_question=self.FLAGS.number_of_words_in_question,
                                        number_of_letters_in_word=self.FLAGS.number_of_letters_in_word,batch_size=self.FLAGS.batch_size,test_size=0.1)

        with tf.variable_scope("QuoraBiRNNModel",initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()
            #self.add_training_step()

            # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
            #params = tf.trainable_variables()
            #gradients = tf.gradients(self.loss, params)
            #self.gradient_norm = tf.global_norm(gradients)
            #clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
            #self.param_norm = tf.global_norm(params)

            # Define optimizer and updates
            # (updates is what you need to fetch in session.run to do a gradient update)
            #self.global_step = tf.Variable(0, name="global_step", trainable=False)
            #opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # you can try other optimizers
            #self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

            # Define savers (for checkpointing) and summaries (for tensorboard)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            self.summaries = tf.summary.merge_all()

    def add_placeholders(self):
        self.q1_ids=tf.placeholder(tf.int32,shape=[None,self.FLAGS.number_of_words_in_question])
        self.q1_mask=tf.placeholder(tf.int32,shape=[None,self.FLAGS.number_of_words_in_question])
        self.q2_ids=tf.placeholder(tf.int32, shape=[None,self.FLAGS.number_of_words_in_question])
        self.q2_mask=tf.placeholder(tf.int32, shape=[None,self.FLAGS.number_of_words_in_question])
        self.labels=tf.placeholder(tf.float32,shape=[None,1])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

    def add_embedding_layer(self,emb_matrix):
        with vs.variable_scope("embeddings"):
            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32,name="emb_matrix")  # shape (91469, embedding_size)

            self.q1_embs=embedding_ops.embedding_lookup(embedding_matrix,self.q1_ids)  # shape (batch_size, context_len, embedding_size)
            self.q2_embs = embedding_ops.embedding_lookup(embedding_matrix,self.q2_ids)  # shape (batch_size, context_len, embedding_size)

    def build_graph(self):
        encoder=BiRNNLayer(name='BiRNNModelEncoder1',hidden_size=self.FLAGS.hidden_size,keep_prob=self.keep_prob)
        zz1,q1_hidden=encoder.build_graph(self.q1_embs,self.q1_mask)
        zz2,q2_hidden=encoder.build_graph(self.q2_embs,self.q2_mask)
        q1_q2_prod=q1_hidden*q2_hidden
        q1_q2_diff=q1_hidden-q2_hidden
        #cos_sim=cosine_similarity(q1_hidden,q2_hidden)
        #blended_reps = tf.concat([q1_hidden, q2_hidden], axis=1)  # (batch_size,2*hidden_size)
        #blended_reps=tf.concat([q1_q2_prod,q1_q2_diff,cos_sim],axis=1)
        blended_reps = tf.concat([q1_q2_prod, q1_q2_diff], axis=1)
        blended_reps=tf.layers.batch_normalization(blended_reps,axis=1)
        DENSE_UNITS=600
        with tf.variable_scope('full_layer1') as scope:
            full_weight1 = tf.get_variable(name='full_layer1_weight', shape=[4 * self.FLAGS.hidden_size,DENSE_UNITS ],
                                           dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.05))
            full_bias1 = tf.get_variable(name='full_layer_bias1', shape=[DENSE_UNITS], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            full_output1= tf.matmul(blended_reps, full_weight1) + full_bias1
            full_weight2 = tf.get_variable(name='full_layer2_weight',
                                           shape=[DENSE_UNITS,1],
                                           dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.05))
            full_bias2 = tf.get_variable(name='full_layer_bias2', shape=[1], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            final_output = tf.matmul(full_output1, full_weight2) + full_bias2
        logits = tf.identity(final_output, name="logits")
        self.final_output = final_output
        self.logits = logits

    def add_loss(self):
        with vs.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate).minimize(self.loss)
            self.predicted = tf.nn.sigmoid(self.final_output)
            self.correct_prediction = tf.equal(tf.round(self.predicted), self.labels,name='correct_prediction')
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')
            tf.summary.scalar('cost', self.loss)


    def run_train_iter(self,sess,q1_ids_batch,q1_mask_batch,q2_ids_batch,q2_mask_batch,labels_batch):
        train_data_feed={
            self.q1_ids:q1_ids_batch,
            self.q1_mask:q1_mask_batch,
            self.q2_ids:q2_ids_batch,
            self.q2_mask:q2_mask_batch,
            self.labels:labels_batch,
            self.keep_prob: (1.0 - self.FLAGS.dropout),
        }
        #sess.run(self.train_step,feed_dict=train_data_feed)
        #output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        #[_, summaries, loss, global_step, param_norm, gradient_norm] = sess.run(output_feed, train_data_feed)
        loss,_,acc=sess.run([self.loss,self.optimizer,self.accuracy],feed_dict=train_data_feed)

        # All summaries in the graph are added to Tensorboard
        #summary_writer.add_summary(summaries, global_step)

        return loss, acc

    def get_loss(self,sess,input_feed):
        output_feed = [self.loss]
        [loss] = sess.run(output_feed, input_feed)
        return loss

    def get_validation_accuracy(self,sess):
        loss_per_batch, batch_lengths = [], []
        accuracy_per_batch=[]
        for q1_ids_batch, q1_mask_batch, q2_ids_batch, q2_mask_batch, labels_batch in self.dataObject.generate_dev_data():
            dev_data_feed = {
                self.q1_ids: q1_ids_batch,
                self.q1_mask: q1_mask_batch,
                self.q2_ids: q2_ids_batch,
                self.q2_mask: q2_mask_batch,
                self.labels: labels_batch,
                self.keep_prob: 1.0,
            }
            #loss = self.get_loss(sess, dev_data_feed)
            loss,acc,pred=sess.run([self.loss,self.accuracy,self.predicted],feed_dict=dev_data_feed)
            #loss,acc=sess.run([self.accuracy])
            curr_batch_size=q1_ids_batch.shape[0]
            loss_per_batch.append(loss * curr_batch_size)
            accuracy_per_batch.append(acc*curr_batch_size)
            batch_lengths.append(curr_batch_size)
        total_num_examples = sum(batch_lengths)
        dev_loss = sum(loss_per_batch) / float(total_num_examples)
        dev_accuracy=sum(accuracy_per_batch)/ float(total_num_examples)
        return dev_loss,dev_accuracy

    def get_test_data(self,sess):
        output=[]
        lineids=[]
        for q1_ids_batch, q1_mask_batch, q2_ids_batch, q2_mask_batch, lineid_batch in self.dataObject.generate_test_data():
            test_data_feed = {
                self.q1_ids: q1_ids_batch,
                self.q1_mask: q1_mask_batch,
                self.q2_ids: q2_ids_batch,
                self.q2_mask: q2_mask_batch,
                self.labels: lineid_batch,
                self.keep_prob: 1.0,
            }
            test_output = sess.run(self.predicted, feed_dict=test_data_feed)
            lineids.extend(lineid_batch.tolist())
            output.extend(test_output.tolist())
        return lineids, output

    def run_epoch(self, sess):
        loss_per_batch, batch_lengths = [], []
        accuracy_per_batch=[]
        for q1_ids_batch, q1_mask_batch, q2_ids_batch, q2_mask_batch, labels_batch in self.dataObject.generate_one_epoch():
            #loss, global_step, param_norm, grad_norm=self.run_train_iter(sess,q1_ids_batch, q1_mask_batch, q2_ids_batch, q2_mask_batch, labels_batch)
            loss, acc=self.run_train_iter(sess,q1_ids_batch, q1_mask_batch, q2_ids_batch, q2_mask_batch, labels_batch)
            curr_batch_size = q1_ids_batch.shape[0]
            loss_per_batch.append(loss * curr_batch_size)
            accuracy_per_batch.append(acc*curr_batch_size)
            batch_lengths.append(curr_batch_size)
        total_num_examples = sum(batch_lengths)
        train_loss = sum(loss_per_batch) / float(total_num_examples)
        train_accuracy=sum(accuracy_per_batch)/float(total_num_examples)
        validation_loss,validation_accuracy = self.get_validation_accuracy(sess)
        return train_loss,train_accuracy,validation_loss,validation_accuracy