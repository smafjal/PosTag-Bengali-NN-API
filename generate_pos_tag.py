#!/usr/bin/python
__author__ = '@smafjal {afjal.sm19@gmail.com}'

import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import precision_recall_fscore_support as score

word_vector_data_path='wordVector/data.pickle'
words_vector_formatDic_path='wordVector/format_data.pickle'

taglist=['N','V','J','D','L','A','C','RD','P','PU','PP']

def chomps(s):
    return s.rstrip('\n')

def get_ascii(s):
    if type(s)!=unicode: return s
    else: return s.encode('utf-8')

def posTagNN(words_vector,postag_model):
    n_classes = len(taglist)
    n_input=900
    n_hidden_1 = int((n_input+n_classes)/2) + 10 # 1st layer num features
    n_hidden_2 = int((n_input+n_classes)/2) + 10 # 2nd layer num features
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    def multilayer_perceptron(_X, _weights, _biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
        return tf.matmul(layer_2, _weights['out']) + _biases['out']
        pass

    pred = multilayer_perceptron(x, weights, biases)
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    y_pred_val=[]

    with tf.Session() as sess:
        sess.run(init)
        # saver.restore(sess,'postag_saved_model/model.ckpt-82888')
        saver.restore(sess,postag_model)
        y_pred=tf.argmax(pred,1)
        y_pred_val=y_pred.eval(feed_dict={x:words_vector},session=sess)
    return y_pred_val

def load_data(path):
    with open(path,'r') as r:
        data=pickle.load(r)
    return data

def read_input_sentence(data_input_path):
    data=[];cnt_sentences=0;
    for line in open(data_input_path):
        words=chomps(line).strip().split(" ")
        for i in range(len(words)):
            word_tuple=(cnt_sentences,i,line)
            data.append(word_tuple)
        cnt_sentences+=1
    return data

def generate_output_sentence(data,format_data,max_len,stotal):
    sentences=[]
    for i in range(stotal):
        one_sen=[]
        for j in range(max_len[i]):
            tup=(i,j)
            one_sen.append(format_data[tup])
            one_sen.append(taglist[data[tup]])
        sentences.append(one_sen)
    return sentences

def write_result(sentences,path):
    with open(path,"w") as w:
        for sen in sentences:
            one_sen=""
            for word in sen:
                one_sen=one_sen+" "+get_ascii(word)
            w.write(one_sen.strip()+"\n")
            pass

def posTag_generator(postag_model_path,output_path):
    stotal=0;
    data=load_data(word_vector_data_path)
    data_list=[]
    for x in data:
        data_list.append((x,data[x]))
        stotal=max(stotal,x[0])

    max_len={};words_vector=[]
    for x in data_list:
        words_vector.append(x[1])
        if x[0][0] in max_len:
            max_len[x[0][0]]=max(max_len[x[0][0]],x[0][1])
        else:max_len[x[0][0]]=x[0][1]

    generated_tag_idx=posTagNN(words_vector,postag_model_path)
    data_idx={};gen_idx=0;
    for x in data_list:
        data_idx[x[0]]=generated_tag_idx[gen_idx]
        gen_idx+=1

    format_data=load_data(words_vector_formatDic_path)
    sentences=generate_output_sentence(data_idx,format_data,max_len,stotal)
    write_result(sentences,output_path)

def main():
    generate_pos_tag(data_input_path)

if __name__=="__main__":
    main()

