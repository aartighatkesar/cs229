from deep_learning_models.model import NeuralLayer
import tensorflow as tf
#from tensorflow import variable_scope as vs
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist

class NeuralNetworkHiddenLayer(NeuralLayer):
    def __init__(self,name,input_size,hidden_size,keep_prop):
        self.name=name
        self.keep_prob=keep_prop
        self.input_size=input_size
        self.hidden_size=hidden_size

    def build_graph(self,values):
        hidden_weight=tf.get_variable(name=self.name+'_hidden_weight',shape=[self.input_size,self.hidden_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05))
        hidden_bias=tf.get_variable(name=self.name+'_hidden_bias',shape=[self.hidden_size],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        output=tf.nn.relu(tf.matmul(values,hidden_weight)+hidden_bias)
        output=tf.nn.dropout(output,self.keep_prob)
        return output

class Convolution2DLayer(NeuralLayer):
    def __init__(self,name,filter_shape,strides,channel_size,pool_size=None):
        self.name=name
        self.filter_shape=filter_shape
        self.strides=strides
        self.channel_size=channel_size
        self.pool_size=pool_size
        #channel_size must = filter[3]
        #filter[2] must=values[3]

    def build_graph(self,values):   #values has shape (batch,X,Y,Z)
        conv_filter=tf.get_variable(name=self.name+'_conv_filter', shape=self.filter_shape,dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05)) #filter is [3,B,Z,channel_size]
        conv=tf.nn.conv2d(values,conv_filter,self.strides,padding='SAME')       #conv has shape (1,X/stride[1],Y/stride[2],channel_size)
        conv_bias=tf.get_variable(name=self.name+'_conv_bias',shape=self.channel_size,dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv_bias_add=tf.nn.bias_add(conv,conv_bias)
        conv_relu=tf.nn.relu(conv_bias_add)
        if self.pool_size is not None:
            context_pool=tf.nn.max_pool(conv_relu,ksize=self.pool_size,strides=self.pool_size,padding='SAME',name=self.name+'_max_pool')
            return context_pool     #conv has shape (1,X/pool_size[1],Y/pool_size[2],channel_size)
        else:
            return conv_relu

class Convolution1DLayer(NeuralLayer):
    def __init__(self,name,filter_shape,strides,channel_size,pool_size=None):
        self.name=name
        self.filter_shape=filter_shape
        self.strides=strides
        self.channel_size=channel_size
        self.pool_size=pool_size
        # channel_size must = filter[2]
        # filter[1] must=values[2]

    def build_graph(self,values):   #values has shape (batch,X,Y)
        conv_filter=tf.get_variable(name=self.name + '_conv_filter', shape=self.filter_shape, dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.05))  # filter is [3,Y,channel_size]
        conv=tf.nn.conv1d(values,conv_filter,self.strides,padding='SAME')  # conv has shape (1,X/stride,channel_size)
        conv_bias=tf.get_variable(name=self.name+'_conv_bias',shape=self.channel_size,dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv_bias_add=tf.nn.bias_add(conv,conv_bias)
        conv_relu=tf.nn.relu(conv_bias_add)
        if self.pool_size is not None:
            context_pool=tf.nn.pool(conv_relu,window_shape=self.pool_size,strides=self.pool_size,pooling_type='MAX',padding='SAME',name=self.name+'_max_pool')
            return context_pool     #context_pool has shape(1,conv_relu[1]/pool_size,channels)
        else:
            return conv_relu  # conv has shape (1,X/pool_size[1],Y/pool_size[2],channel_size)

class LSTMLayer(NeuralLayer):
    def __init__(self,name,hidden_size,keep_prop,num_layers=1):
        self.name=name
        self.hidden_size=hidden_size
        self.keep_prob=keep_prop
        if num_layers==1:
            self.rnn_cell=self.create_one_cell()
        else:
            self.rnn_cell=tf.contrib.rnn.MultiRNNCell([self.create_one_cell() for _ in range(num_layers)],state_is_tuple=True)

    def create_one_cell(self):
        cell=tf.contrib.rnn.LSTMCell(self.hidden_size)
        cell=tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=self.keep_prob)
        return cell

    def build_graph(self,values,masks):
        input_lens=tf.reduce_sum(masks, reduction_indices=1)
        val,state=tf.nn.dynamic_rnn(cell=self.rnn_cell,inputs=values,sequence_length=input_lens,dtype=tf.float32)   #val is (batch,num_words,hidden_size)
        final_value=tf.gather(val,int(val.get_shape()[1]) - 1,axis=1,name="lstm_state")
        return final_value          #batch,hidden_size

class BiRNNLayer(NeuralLayer):
    def __init__(self,name,hidden_size,keep_prob,typeOfRNN='LSTM'):
        self.name=name
        self.hidden_size=hidden_size
        self.keep_prob=keep_prob
        if typeOfRNN=='GRU':
            self.rnn_cell_fw = tf.contrib.rnn.GRUCell(self.hidden_size)
            self.rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
            self.rnn_cell_bw = tf.contrib.rnn.GRUCell(self.hidden_size)
            self.rnn_cell_bw = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)
        else:
            self.rnn_cell_fw=tf.contrib.rnn.LSTMCell(self.hidden_size)
            self.rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
            self.rnn_cell_bw=tf.contrib.rnn.LSTMCell(self.hidden_size)
            self.rnn_cell_bw = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self,inputs,masks):
        with vs.variable_scope(self.name):
            input_lens = tf.reduce_sum(masks, reduction_indices=1)  # shape (batch_size)
            (fw_out,bw_out),(fw_final_state,bw_final_state)=tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw,self.rnn_cell_bw,inputs,input_lens,dtype=tf.float32)
            fw_out_last=fw_final_state.h
            bw_out_last=bw_final_state.h
            out=tf.concat([fw_out,bw_out],2)
            out_last=tf.concat([fw_out_last,bw_out_last],1)
            out=tf.nn.dropout(out,self.keep_prob)
            out_last= tf.nn.dropout(out_last,self.keep_prob)
            return out,out_last

class SimpleSoftmax(NeuralLayer):
    def __init__(self):
        pass

    def build_graph(self,inputs,masks):
        with vs.variable_scope("SimpleSoftmaxLayer"):
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)
            return masked_logits, prob_dist

class BasicAttn(NeuralLayer):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

class RNNEncoder(NeuralLayer):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = tf.contrib.rnn.GRUCell(self.hidden_size)
        self.rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = tf.contrib.rnn.GRUCell(self.hidden_size)
        self.rnn_cell_bw = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out