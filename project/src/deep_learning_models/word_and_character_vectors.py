from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np
import os

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1

#these are the lines in the glove and fasttext files
#GLOVE_VOCAB_SIZE=5000
#GLOVE_FILENAME='glove_small.txt'
GLOVE_VOCAB_SIZE=1917494
GLOVE_FILENAME='glove.42B.300d.txt'
#GLOVE_VOCAB_SIZE=400000
#GLOVE_FILENAME='glove.6B.300d.txt'
GLOVE_DIMENSION=300
FASTTEXT_VOCAB_SIZE=2000000
FASTTEXT_FILENAME='crawl-300d-2M.vec'
FASTTEXT_DIMENSION=300
#CHAR_VOCAB_SIZE=65
CHAR_FILENAME='char_dim_wordSize_word_size.txt'
#CHAR_DIMENSION=128
#CHAR_WORD_SAMPLING=5
GLOVE_TWITTER_VOCAB_SIZE=1193514
GLOVE_TWITTER_FILENAME='glove.twitter.27B.200d.txt'
GLOVE_TWITTER_DIMENSION=200
WORD2VEC_VOCAB_SIZE=3000000
WORD2VEC_DIMENSION=300
WORD2VEC_FILENAME='GoogleNews-vectors-negative300.txt'


def get_char(data_file_path,CHAR_DIMENSION=128,CHAR_WORD_SAMPLING=5,CHAR_VOCAB_SIZE=65):
    filename=CHAR_FILENAME
    filename=filename.replace('dim',str(CHAR_DIMENSION))
    filename=filename.replace('word_size',str(CHAR_WORD_SAMPLING))
    path=os.path.join(data_file_path,filename)
    return get_character_embeddings(path,CHAR_VOCAB_SIZE,CHAR_DIMENSION)

def get_character_embeddings(datafile,vocab_size,dimension):
    print("Loading vectors from file: %s" % datafile)

    dim = dimension
    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), dim))
    char2id = {}
    id2char = {}

    random_init_padding=False
    if random_init_padding:
        emb_matrix[0:1, :] = np.random.randn(1, dim)

    use_existing_unknown=True
    random_init=False
    if random_init:
        emb_matrix[1:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB)-1, dim)
        #emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        char2id[word] = idx
        id2char[idx] = word
        idx += 1

    # go through vecs
    with open(datafile, 'r', encoding="utf8") as fh:
        for line in fh:
            line = line.rstrip().split("\t")
            char = line[0]
            vector = list(map(float, line[1:]))
            emb_matrix[idx, :] = vector
            char2id[char] = idx
            id2char[idx] = char
            idx += 1

    if use_existing_unknown:
        emb_matrix[char2id[_UNK]]=emb_matrix[char2id['UNKNOWN_TOKEN']]


    final_vocab_size =  vocab_size + len(_START_VOCAB)
    assert len(char2id) == final_vocab_size
    assert len(id2char) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, char2id, id2char

#get twitter vectors.
def get_glove_twitter(data_file_path):
    path=os.path.join(data_file_path,GLOVE_TWITTER_FILENAME)
    return get_word_embeddings(path,GLOVE_TWITTER_VOCAB_SIZE,GLOVE_TWITTER_DIMENSION)

#get google news vectors
def get_word2vec(data_file_path):
    path=os.path.join(data_file_path,WORD2VEC_FILENAME)
    return get_word_embeddings(path,WORD2VEC_VOCAB_SIZE,WORD2VEC_DIMENSION)

#get glove vectors. need to pass just the data path location
#adds glove.42B.300d to path
def get_glove(data_file_path):
    path=os.path.join(data_file_path,GLOVE_FILENAME)
    return get_word_embeddings(path,GLOVE_VOCAB_SIZE,GLOVE_DIMENSION)

#gets fasttext vectors. need to pass just the data path location
#adds glove.42B.300d to path
def get_fasttext(data_file_path):
    path = os.path.join(data_file_path, FASTTEXT_FILENAME)
    return get_word_embeddings(path, FASTTEXT_VOCAB_SIZE,FASTTEXT_DIMENSION)

def get_word_embeddings(datafile,vocab_size,dimension):
    """Reads from the data file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      datafile: path to data file
      vocab_size: size of vocabulary
      dimension: vector dimension

    Returns:
      emb_matrix: Numpy array shape (vocab_size, dimension) containing vector embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """
    print ("Loading vectors from file: %s" % datafile)
    dim = dimension

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), dim))
    word2id = {}
    id2word = {}

    random_init = False
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(datafile, 'r', encoding="utf8") as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word

