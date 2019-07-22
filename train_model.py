import os
import sys
import codecs
import gensim

class MySentences(object):
     def __init__(self, dirname):
         self.dirname = dirname

     def __iter__(self):
         for fname in os.listdir(self.dirname):
             for line in codecs.open(os.path.join(self.dirname, fname), 'r', 'utf-8'):
                 yield line.split()

dirData = './data_train/dongdu_tokenize/'
pathModelBin='/home/huutinh/GMO/Deep_learning/model/vnw2v.bin'
pathModelTxt='/home/huutinh/GMO/Deep_learning/model/vnw2v.txt'

if __name__ == '__main__':
    sentences = MySentences(dirData) # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, size=300, window=10, min_count=10, sample=0.0001, workers=4, sg=0, negative=10, cbow_mean=1, iter=5)
    model.wv.save_word2vec_format(pathModelBin, fvocab=None, binary=True)
    model.wv.save_word2vec_format(pathModelTxt, fvocab=None, binary=False)
