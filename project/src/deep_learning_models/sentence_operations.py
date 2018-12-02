from deep_learning_models.word_and_character_vectors import PAD_ID,UNK_ID
import re
from nltk import word_tokenize

def sentence_to_word_ids(sentence,word2id,stopwords=set()):
    """
    convert tokenized sentence into word indices
    any word not present gets converted to unknown
    """
    tokens=process_text(sentence)
    word_ids=[]
    for w in tokens:
        if w not in stopwords:
            word_ids.append(word2id.get(w,UNK_ID))
    return tokens,word_ids

def sentence_to_char_ids(sentence,char2id):
    """
    convert tokenized sentence into char indices
    any char not present gets converted to unknown
    """
    tokens=process_text(sentence)
    char_ids=[]
    for w in tokens:
        char_word_ids=[]
        for c in w:
            char_word_ids.append(char2id.get(c,UNK_ID))
        char_ids.append(char_word_ids)

    return tokens,char_ids

def tokens_to_word_ids(tokens,word2id):
    """
    convert tokenized sentence into word indices
    any word not present gets converted to unknown
    """
    word_ids=[]
    for w in tokens:
        word_ids.append(word2id.get(w,UNK_ID))

    return word_ids

def tokens_to_char_ids(tokens,char2id):
    """
    convert tokenized sentence into char indices
    any char not present gets converted to unknown
    """
    char_ids=[]
    for w in tokens:
        char_word_ids=[]
        for c in w:
            char_word_ids.append(char2id.get(c,UNK_ID))
        char_ids.append(char_word_ids)

    return char_ids


def sentence_to_word_and_char_token_ids(sentence,word2id,char2id):
    """
    convert tokenized sentence into word indices
    any word not present gets converted to unknown
    """
    tokens=split_by_whitespace(sentence)
    word_ids=[]
    char_ids=[]
    for w in tokens:

        word_ids.append(word2id.get(w,UNK_ID))
        char_word_ids=[]
        for c in w:
            char_word_ids.append(char2id.get(c,UNK_ID))
        char_ids.append(char_word_ids)

    return tokens,word_ids,char_ids

def pad_words(word_list,pad_size):
    return word_list+[PAD_ID]*(pad_size-len(word_list))

#---------------
#need to fix this!!!
#--------------
def pad_characters(char_array,pad_size,word_pad_size):
    if len(char_array)<pad_size:
        char_array=char_array+[[PAD_ID]]*(pad_size-len(char_array))
    for i,item in enumerate(char_array):
        if len(item)<word_pad_size:
            char_array[i]=char_array[i]+[PAD_ID]*(word_pad_size-len(item))
        if len(item) > word_pad_size:
            char_array[i] = item[:word_pad_size]
    return char_array

def convert_ids_to_word_vectors(word_ids,emb_matrix_word):
    retval=[]
    for id in word_ids:
        retval.append(emb_matrix_word[id].tolist())
    return retval

def convert_ids_to_char_vectors(char_ids,emb_matrix_char):
    retval=[]
    for word_rows in char_ids:
        row_val=[]
        for c in word_rows:
            row_val.append(emb_matrix_char[c].tolist())
        retval.append(row_val)
    return retval

def get_ids_and_vectors(text,word2id,char2id,word_embed_matrix,char_embed_matrix,review_length,word_length,discard_long):
    tokens, word_ids, char_ids = sentence_to_word_and_char_token_ids(text, word2id, char2id)
    if len(tokens) > review_length:
        if discard_long:
            return None,None,None
        else:
            tokens = tokens[:review_length]
            word_ids = word_ids[:review_length]
            char_ids = char_ids[:review_length]
    word_ids = pad_words(word_ids, review_length)
    char_ids = pad_characters(char_ids, review_length, word_length)
    word_ids_to_vectors = convert_ids_to_word_vectors(word_ids, word_embed_matrix)
    char_ids_to_vectors = convert_ids_to_char_vectors(char_ids, char_embed_matrix)
    return word_ids,word_ids_to_vectors,char_ids_to_vectors


def process_text(text,stopwords=set()):
    return_tokens=[]
    words=word_tokenize(text.lower())
    p = re.compile('[a-zA-Z]+')
    for word in words:
        if word not in stopwords:
            temp=p.match(word)
            if temp is not None:
                return_tokens.append(p.match(word).group(0))
    return return_tokens