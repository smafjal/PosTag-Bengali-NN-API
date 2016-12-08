#!/usr/bin/python
__author__ = '@smafjal {afjal.sm19@gmail.com}'

import argparse
import data_reader
import generate_pos_tag as posTag

def get_args():
    parser = argparse.ArgumentParser(description='Bengali Parts of Speech Tagging by using Neural Network')
    parser.add_argument('-in', '--input', type=str,
        help='input file name', required=False,default='input/input.txt')
    parser.add_argument('-out', '--output', type=str,
        help='output file name', required=False, default='output/output.txt')
    parser.add_argument('-posm', '--postagmodel', type=str,
        help='Pos Tag trained model file path', required=False, default='posTagModel/model.ckpt-82888')
    parser.add_argument('-wem', '--wordembeddingmodel', type=str,
        help='Word Embedding model file path', required=False, default='wordEmbeddingModel/pipilika_bd_big_news.model.bin')

    args = parser.parse_args()
    input_path=args.input
    output_path=args.output
    postag_model_path=args.postagmodel
    word_embedding_model_path=args.wordembeddingmodel
    return input_path,output_path,word_embedding_model_path,postag_model_path

def main():
    input_path,output_path,word_embedding_model_path,postag_model_path=get_args()
    print "\n\nINPUT: ",input_path
    print "OUTPUT: ",output_path
    print "Word Embedding Model: ",word_embedding_model_path
    print "Pos Tag Model: ",postag_model_path
    print "---------------------------"

    # data,format_dic=data_reader(input_path,word_embedding_model_path)
    print "Data Reading Completed!!"

    posTag.posTag_generator(postag_model_path,output_path)
    print "Your output saved on --->",output_path,"\n"

# Run without fear
if __name__=="__main__":
    main()

# If you have knowledge,
# let others light their candles in it.
# -----Margaret Fuller

