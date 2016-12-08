#!/usr/bin/python
__author__ = '@smafjal {afjal.sm19@gmail.com}'

from sets import Set
import numpy as np
import os,codecs
import pickle
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def chomps(s):
    return s.rstrip('\n')

# Get Uncode of string
def get_unicode(input):
    input=chomps(input)
    if type(input) == unicode: return input
    else: return input.decode('utf-8')

def get_ascii(input):
    if type(input) != unicode:return input
    else: return input.encode("utf-8")

def generate_vector(words,word_idx,model):
    end_pad=model[get_unicode("<PAD>")]
    embedding_vector=[]; taken=0
    not_found_word_cnt=0
    for i in range(word_idx-1,len(words)):
        taken+=1
        if taken>3: break
        word=get_unicode(words[i])
        word_vec=end_pad
        if word in model:
            word_vec=model[word]
        else: not_found_word_cnt+=1
        embedding_vector.append(word_vec)

    print not_found_word_cnt,"words not found on embedding model"
    vlen=len(embedding_vector)
    for i in range(min(3,3-vlen)):
        embedding_vector.append(end_pad)
    embedding_vector=np.reshape(embedding_vector,-1)
    return embedding_vector

def load_embedding_model(path):
    print "Big Size of Embedding Model is loading. Plz be forbearing"
    model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    print "Model loading Complete!!"
    return model

def write_format_data(data,path):
    with open(path,"w") as w:
        for x in data:
            wstr=""+str(x[0])+" "+str(x[1])+" "+data[x]
            w.write(wstr+"\n")

def write_to_file(data,format_data):
    with open("wordVector/data.pickle",'wb') as w:
        pickle.dump(data,w)
    with open("wordVector/format_data.pickle",'wb') as w:
        pickle.dump(format_data,w)

def read_data(data_input_path,embedding_model_path):
    embedding_model=load_embedding_model(embedding_model_path)
    data={};format_dic={};cnt_sentences=0
    for line in open(data_input_path):
        words=chomps(line).strip().split(" ")
        for i in range(len(words)):
            word_vector=generate_vector(words,i,embedding_model)
            word_tuple=(cnt_sentences,i)
            data[word_tuple]=word_vector
            format_dic[word_tuple]=words[i]
            pass
        cnt_sentences+=1
    write_to_file(data,format_dic)
    pass

