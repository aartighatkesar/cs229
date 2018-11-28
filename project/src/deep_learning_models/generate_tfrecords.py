from deep_learning_models.word_and_character_vectors import get_char,get_glove,PAD_ID,UNK_ID,get_fasttext
from deep_learning_models.sentence_operations import sentence_to_word_ids,pad_words,convert_ids_to_word_vectors
import tensorflow as tf
import GlobalParameters
from sklearn.model_selection import train_test_split
import numpy as np

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def generate_glove_tf_record(word2id,glove_matrix,num_words,test_size=0.1):
    vars=GlobalParameters.GlobalVariables()
    train_data,val_data=train_test_split(vars.train,test_size=test_size)
    test_data=vars.test
    train_file='train_glove.tfrecords'
    val_file='val_glove.tfrecords'
    test_file='test_glove.tfrecords'
    count=0
    print("Parsing train file")
    writer = tf.python_io.TFRecordWriter(train_file)
    for i,row in train_data.iterrows():
        if count%10000==0:
            print("Done "+str(count) +" records")
        count+=1
        q1=row['question1']
        q2=row['question2']
        is_duplicate=row['is_duplicate']
        q1_tokens,q1_word_ids=sentence_to_word_ids(q1.lower(),word2id)
        if(len(q1_tokens))>num_words:
            q1_tokens=q1_tokens[:num_words]
            q1_word_ids=q1_word_ids[:num_words]
        q1_word_ids=pad_words(q1_word_ids,num_words)
        q1_word_ids_to_vectors=convert_ids_to_word_vectors(q1_word_ids,glove_matrix)
        q1_word_mask=[int(w!=PAD_ID) for w in q1_word_ids]
        q1_word_ids=np.array(q1_word_ids)
        q1_word_ids_to_vectors=np.array(q1_word_ids_to_vectors)
        q1_word_mask=np.array(q1_word_mask)
        q2_tokens,q2_word_ids=sentence_to_word_ids(q2.lower(), word2id)
        if (len(q2_tokens)) > num_words:
            q2_tokens = q2_tokens[:num_words]
            q2_word_ids = q2_word_ids[:num_words]
        q2_word_ids=pad_words(q2_word_ids,num_words)
        q2_word_ids_to_vectors=convert_ids_to_word_vectors(q2_word_ids,glove_matrix)
        q2_word_mask=[int(w!=PAD_ID) for w in q2_word_ids]
        q2_word_ids=np.array(q2_word_ids)
        q2_word_ids_to_vectors=np.array(q2_word_ids_to_vectors)
        q2_word_mask=np.array(q2_word_mask)
        q1_word_ids_to_vectors=np.reshape(q1_word_ids_to_vectors,(num_words*300,))
        q2_word_ids_to_vectors=np.reshape(q2_word_ids_to_vectors,(num_words*300,))
        feature_dict={
            #'q1_word_ids':float_feature(q1_word_ids),
            'q1_word_vectors':float_feature(q1_word_ids_to_vectors),
            'q1_word_mask':float_feature(q1_word_mask),
            #'q2_word_ids':float_feature(q2_word_ids),
            'q2_word_vectors':float_feature(q2_word_ids_to_vectors),
            'q2_word_mask':float_feature(q2_word_mask),
            'is_duplicate':float_feature([is_duplicate])
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        writer.write(example.SerializeToString())
    writer.flush()
    writer.close()
    print("Parsing train file...done")
    count=0
    print("Parsing val file")
    writer = tf.python_io.TFRecordWriter(val_file)
    for i, row in val_data.iterrows():
        if count%10000==0:
            print("Done "+str(count) +" records")
        count+=1
        q1 = row['question1']
        q2 = row['question2']
        is_duplicate = row['is_duplicate']
        q1_tokens, q1_word_ids = sentence_to_word_ids(q1.lower(), word2id)
        if (len(q1_tokens)) > num_words:
            q1_tokens = q1_tokens[:num_words]
            q1_word_ids = q1_word_ids[:num_words]
        q1_word_ids = pad_words(q1_word_ids, num_words)
        q1_word_ids_to_vectors = convert_ids_to_word_vectors(q1_word_ids, glove_matrix)
        q1_word_mask = [int(w != PAD_ID) for w in q1_word_ids]
        q1_word_ids = np.array(q1_word_ids)
        q1_word_ids_to_vectors = np.array(q1_word_ids_to_vectors)
        q1_word_mask = np.array(q1_word_mask)
        q2_tokens, q2_word_ids = sentence_to_word_ids(q2.lower(), word2id)
        if (len(q2_tokens)) > num_words:
            q2_tokens = q2_tokens[:num_words]
            q2_word_ids = q2_word_ids[:num_words]
        q2_word_ids = pad_words(q2_word_ids, num_words)
        q2_word_ids_to_vectors = convert_ids_to_word_vectors(q2_word_ids, glove_matrix)
        q2_word_mask = [int(w != PAD_ID) for w in q2_word_ids]
        q2_word_ids = np.array(q2_word_ids)
        q2_word_ids_to_vectors = np.array(q2_word_ids_to_vectors)
        q2_word_mask = np.array(q2_word_mask)
        q1_word_ids_to_vectors = np.reshape(q1_word_ids_to_vectors, (num_words * 300,))
        q2_word_ids_to_vectors = np.reshape(q2_word_ids_to_vectors, (num_words * 300,))
        feature_dict = {
            #'q1_word_ids':float_feature(q1_word_ids),
            'q1_word_vectors':float_feature(q1_word_ids_to_vectors),
            'q1_word_mask':float_feature(q1_word_mask),
            #'q2_word_ids':float_feature(q2_word_ids),
            'q2_word_vectors':float_feature(q2_word_ids_to_vectors),
            'q2_word_mask':float_feature(q2_word_mask),
            'is_duplicate':float_feature([is_duplicate])
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        writer.write(example.SerializeToString())
    writer.flush()
    writer.close()
    count=0
    print("Parsing test file")
    writer = tf.python_io.TFRecordWriter(test_file)
    for i, row in test_data.iterrows():
        if count%10000==0:
            print("Done "+str(count) +" records")
        count+=1
        q1 = row['question1']
        q2 = row['question2']
        is_duplicate = row['test_id']
        q1_tokens, q1_word_ids = sentence_to_word_ids(q1.lower(), word2id)
        if (len(q1_tokens)) > num_words:
            q1_tokens = q1_tokens[:num_words]
            q1_word_ids = q1_word_ids[:num_words]
        q1_word_ids = pad_words(q1_word_ids, num_words)
        q1_word_ids_to_vectors = convert_ids_to_word_vectors(q1_word_ids, glove_matrix)
        q1_word_mask = [int(w != PAD_ID) for w in q1_word_ids]
        q1_word_ids = np.array(q1_word_ids)
        q1_word_ids_to_vectors = np.array(q1_word_ids_to_vectors)
        q1_word_mask = np.array(q1_word_mask)
        q2_tokens, q2_word_ids = sentence_to_word_ids(q2.lower(), word2id)
        if (len(q2_tokens)) > num_words:
            q2_tokens = q2_tokens[:num_words]
            q2_word_ids = q2_word_ids[:num_words]
        q2_word_ids = pad_words(q2_word_ids, num_words)
        q2_word_ids_to_vectors = convert_ids_to_word_vectors(q2_word_ids, glove_matrix)
        q2_word_mask = [int(w != PAD_ID) for w in q2_word_ids]
        q2_word_ids = np.array(q2_word_ids)
        q2_word_ids_to_vectors = np.array(q2_word_ids_to_vectors)
        q2_word_mask = np.array(q2_word_mask)
        q1_word_ids_to_vectors = np.reshape(q1_word_ids_to_vectors, (num_words * 300,))
        q2_word_ids_to_vectors = np.reshape(q2_word_ids_to_vectors, (num_words * 300,))
        feature_dict = {
            #'q1_word_ids':float_feature(q1_word_ids),
            'q1_word_vectors':float_feature(q1_word_ids_to_vectors),
            'q1_word_mask':float_feature(q1_word_mask),
            #'q2_word_ids':float_feature(q2_word_ids),
            'q2_word_vectors':float_feature(q2_word_ids_to_vectors),
            'q2_word_mask':float_feature(q2_word_mask),
            'is_duplicate':float_feature([is_duplicate])
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        writer.write(example.SerializeToString())
    writer.flush()
    writer.close()
    print("Parsing test file...done")

ML_DATA_FILES="c:\\Users\\tihor\\Documents\\ml_data_files\\"
word_embed_matrix,word2id,id2word=get_glove(ML_DATA_FILES)
#word_embed_matrix,word2id,id2word=get_fasttext(ML_DATA_FILES)
num_words=30
generate_glove_tf_record(word2id,word_embed_matrix,num_words)


def _parse_function(example_proto):
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

dataset_train=tf.data.TFRecordDataset('train_glove.tfrecords').shuffle(buffer_size=100000).map(_parse_function, num_parallel_calls=5).repeat(1).batch(10)
iterator_train=dataset_train.make_initializable_iterator()
next_element_train=iterator_train.get_next()

dataset_val=tf.data.TFRecordDataset('val_glove.tfrecords').shuffle(buffer_size=100000).map(_parse_function, num_parallel_calls=5).repeat(1).batch(10)
iterator_val=dataset_val.make_initializable_iterator()
next_element_val=iterator_val.get_next()

dataset_test=tf.data.TFRecordDataset('test_glove.tfrecords').map(_parse_function, num_parallel_calls=5).repeat(1).batch(10)
iterator_test=dataset_test.make_initializable_iterator()
next_element_test=iterator_test.get_next()

sess=tf.Session()
print("-------Running Train--------")
for i in range(1):
    sess.run(iterator_train.initializer)
    while True:
        try:
            a,b,c,d,e=sess.run(next_element_train)
            print(a)
            print(b)
            print(c)
            print(d)
            print(e)
            print("--------------------------------")
            break
        except tf.errors.OutOfRangeError:
            print("Reached end of batch!!!")
            break
print("-------Running Val--------")
for i in range(1):
    sess.run(iterator_val.initializer)
    while True:
        try:
            a,b,c,d,e=sess.run(next_element_val)
            print(a)
            print(b)
            print(c)
            print(d)
            print(e)
            print("--------------------------------")
            break
        except tf.errors.OutOfRangeError:
            print("Reached end of batch!!!")
            break
print("-------Running Test--------")
for i in range(1):
    sess.run(iterator_test.initializer)
    while True:
        try:
            a,b,c,d,e=sess.run(next_element_test)
            print(a)
            print(b)
            print(c)
            print(d)
            print(e)
            print("--------------------------------")
            break
        except tf.errors.OutOfRangeError:
            print("Reached end of batch!!!")
            break