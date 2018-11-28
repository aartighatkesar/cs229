from deep_learning_models.model import Model
from deep_learning_models.modules import NeuralNetworkHiddenLayer
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

def _parse_function(example_proto):
    num_words=30
    features={
        #'q1_word_ids':tf.FixedLenFeature([num_words],tf.float32),
        'q1_word_vectors': tf.FixedLenFeature([num_words*300], tf.float32),
        'q1_word_mask': tf.FixedLenFeature([num_words], tf.float32),
        #'q2_word_ids': tf.FixedLenFeature([num_words], tf.int64),
        'q2_word_vectors': tf.FixedLenFeature([num_words * 300], tf.float32),
        'q2_word_mask': tf.FixedLenFeature([num_words], tf.float32),
        'is_duplicate':tf.FixedLenFeature([1],tf.float32)
    }
    parsed_features=tf.parse_single_example(example_proto,features)
    q1_word_vectors_reshaped=tf.reshape(parsed_features['q1_word_vectors'],[num_words,300])
    q2_word_vectors_reshaped=tf.reshape(parsed_features['q2_word_vectors'], [num_words, 300])
    #return parsed_features['q1_word_ids'],q1_word_vectors_reshaped,parsed_features['q1_word_mask'],parsed_features['q2_word_ids'],q2_word_vectors_reshaped,parsed_features['q2_word_mask'],parsed_features['is_duplicate']
    return q1_word_vectors_reshaped,parsed_features['q1_word_mask'],q2_word_vectors_reshaped,parsed_features['q2_word_mask'],parsed_features['is_duplicate']

class QuoraVanillaNeuralNetworkModel(Model):
    def __init__(self,FLAGS,sess):
        self.FLAGS=FLAGS
        self.dataset_train=tf.data.TFRecordDataset('train_glove.tfrecords').shuffle(buffer_size=100000).map(_parse_function, num_parallel_calls=5).repeat(1).batch(FLAGS.batch_size)
        self.iterator_train=self.dataset_train.make_initializable_iterator()
        self.dataset_val=tf.data.TFRecordDataset('val_glove.tfrecords').shuffle(buffer_size=100000).map(_parse_function,num_parallel_calls=5).repeat(1).batch(FLAGS.batch_size)
        self.iterator_val=self.dataset_val.make_initializable_iterator()
        self.dataset_test=tf.data.TFRecordDataset('test_glove.tfrecords').map(_parse_function, num_parallel_calls=5).repeat(1).batch(FLAGS.batch_size)
        self.iterator_test=self.dataset_test.make_initializable_iterator()
        sess.run(self.iterator_train.initializer)
        sess.run(self.iterator_val.initializer)
        sess.run(self.iterator_test.initializer)
        self.handle_train = sess.run(self.iterator_train.string_handle())
        self.handle_dev = sess.run(self.iterator_val.string_handle())
        self.handle_test = sess.run(self.iterator_test.string_handle())
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element=tf.data.Iterator.from_string_handle(self.handle,self.dataset_train.output_types,self.dataset_train.output_shapes).get_next()
        with tf.variable_scope("QuoraModelNeuralNetwork",initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,uniform=True)):
            self.add_placeholders()
            self.build_graph()
            self.add_loss()
            self.add_training_step()

    def add_placeholders(self):
        #self.q1_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.number_of_words_in_question,self.FLAGS.word_embedding_size])
        #self.q1_mask=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.number_of_words_in_question])
        #self.q2_words=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.number_of_words_in_question,self.FLAGS.word_embedding_size])
        #self.q2_mask=tf.placeholder(dtype=tf.float32,shape=[None,self.FLAGS.number_of_words_in_question])
        #self.is_duplicate=tf.placeholder(dtype=tf.float32,shape=[None,1])
        #self.q1_word_ids,self.q1_words,self.q1_mask,self.q2_word_ids,self.q2_words,self.q2_mask,self.is_duplicate=self.next_element
        self.q1_words, self.q1_mask, self.q2_words, self.q2_mask, self.is_duplicate = self.next_element
        self.keep_prob=tf.placeholder_with_default(1.0,shape=())

    def build_graph(self):
        HIDDEN_LAYER1_SIZE=2048
        HIDDEN_LAYER2_SIZE=2048
        HIDDEN_LAYER3_SIZE=2048

        q1_words_sum=tf.reduce_sum(self.q1_words,1)   #(batch,word_embedding_size)
        q1_mask_sum=tf.reduce_sum(self.q1_mask,1,keepdims=True) +tf.constant(0.0001,dtype=tf.float32,shape=())  #(batch,1)
        input_vector1=q1_words_sum/q1_mask_sum  #(batch,word_embedding_size)

        q2_words_sum=tf.reduce_sum(self.q2_words, 1)  # (batch,word_embedding_size)
        q2_mask_sum=tf.reduce_sum(self.q2_mask, 1, keepdims=True) + tf.constant(0.0001, dtype=tf.float32,shape=())  # (batch,1)
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

    def get_validation_accuracy(self,sess):
        validation_accuracy=0.0
        batch_count=0
        sess.run(self.iterator_val.initializer)
        while True:
            try:
                val_acc_batch=sess.run([self.accuracy],feed_dict={self.handle:self.handle_dev})
                validation_accuracy+=val_acc_batch[0]
                batch_count+=1
            except tf.errors.OutOfRangeError:
                #print("Reached end of batch!!!")
                break
        validation_accuracy/=batch_count
        return validation_accuracy

    def get_test_data(self,sess):
        test_results=[]
        sess.run(self.iterator_test.initializer)
        while True:
            try:
                test_output=sess.run(self.final_output,feed_dict={self.handle:self.handle_test})
                test_results.extend(test_output.tolist())
            except tf.errors.OutOfRangeError:
                #print("Reached end of batch!!!")
                break
        return test_results

    def run_epoch(self,sess):
        sess.run(self.iterator_train.initializer)
        while True:
            try:
                sess.run(self.train_step,feed_dict={self.handle:self.handle_train})
            except tf.errors.OutOfRangeError:
                #print("Reached end of batch!!!")
                break
        validation_accuracy=self.get_validation_accuracy(sess)
        return validation_accuracy