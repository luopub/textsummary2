# -*- coding:utf-8 -*-
# Created by Kevin Luo on 2019-12-15

"""
This file is for data processing
1. Load train and test data from csv file
2. Tonkenize data
3. Remove stop words
4. Get embedding
5. Create dataset
"""

# import modules
import os
import sys
import pathlib
import pandas as pd
import numpy as np
import pdb
from gensim.models.word2vec import LineSentence, Word2Vec

from utils.config import *
from utils.multi_proc_utils import *
from utils.data_loader import *
from utils.file_utils import *

class BaseDataSet():
  def __init__(self, train_csv="", test_csv="", x_cols=[], y_cols=[]):
    self.train_csv = train_csv
    self.test_csv = test_csv
    self.x_cols = x_cols
    self.y_cols = y_cols
    
    self.train_df = None
    self.test_df = None
    self.merged_df = None
    
    self.train_y_max_len = 0
    self.Y_max_len = 0
    
    self.X_max_len = 0
		
    self.train_X = None
    self.train_Y = None
    self.test_X = None

    self.train_ids_x = None
    self.train_ids_y = None
    self.test_ids_x = None
		
    self.wv_model = None
		
    self.vocab = None
    self.reverse_vocab = None
		
    self.embedding_matrix = None

    self.WORD_START = '<START>'
    self.WORD_STOP = '<STOP>'
    self.WORD_UNK = '<UNK>'
    self.WORD_PAD = '<PAD>'

    self.token_id_start = 0
    self.token_id_stop = 0
    self.token_id_unk = 0
    self.token_id_pad = 0
    
    self.vocab_word_list = []
    self.vocab_max_id = 0
		
  def prepare_data(self, force_build = True):
    # pdb.set_trace()

    if force_build or not os.access(train_seg_path, os.F_OK):
      # read in the raw file
      self.train_df = pd.read_csv(self.train_csv)
      self.test_df = pd.read_csv(self.test_csv)
      print('Orgin train data size {},test data size {}'.format(len(self.train_df), len(self.test_df)))
      
      # drop null records
      self.train_df.dropna(subset=self.x_cols+self.y_cols, how='any', inplace=True)
      self.test_df.dropna(subset=self.x_cols, how='any', inplace=True)
      print('Dropna train data size {},test data size {}'.format(len(self.train_df), len(self.test_df)))
        
      # tokenize and split and remove other symbols, then save to file
      self.train_df = parallelize(self.train_df, sentences_proc)
      self.test_df = parallelize(self.test_df, sentences_proc)

      # drop null records again
      self.train_df.replace(to_replace={'':None}, inplace=True)
      self.test_df.replace(to_replace={'':None}, inplace=True)
      self.train_df.dropna(subset=self.x_cols+self.y_cols, how='any', inplace=True)
      self.test_df.dropna(subset=self.x_cols, how='any', inplace=True)
      
      # Save to file
      self.train_df.to_csv(train_seg_path, index=None, header=True)
      self.test_df.to_csv(test_seg_path, index=None, header=True)
      print("File saved: ", train_seg_path, test_seg_path)
    else:
      self.train_df = pd.read_csv(train_seg_path)
      self.test_df = pd.read_csv(test_seg_path)
      print("File read: ", train_seg_path, test_seg_path)
    print('sentences_proced train data size {},test data size {}'.format(len(self.train_df), len(self.test_df)))
    
    if force_build or not os.access(merger_seg_path, os.F_OK):
      # Merge all text together to for embedding
      self.train_df['merged'] = self.train_df[self.x_cols+self.y_cols].apply(lambda x: ' '.join(x), axis=1)
      self.test_df['merged'] = self.test_df[self.x_cols].apply(lambda x: ' '.join(x), axis=1)
      self.merged_df = pd.concat([self.train_df[['merged']], self.test_df[['merged']]], axis=0)

      # Remove the merged column, they are not used anymore
      self.train_df = self.train_df.drop(['merged'], axis=1)
      self.test_df = self.test_df.drop(['merged'], axis=1)

      # save merged to file, this is used later for 
      self.merged_df.to_csv(merger_seg_path, index=None, header=False)
      print("File saved: ", merger_seg_path)

      print('train data size {},test data size {},merged_df data size {}'.format(len(self.train_df), len(self.test_df),len(self.merged_df)))
    else:
      self.merged_df = pd.read_csv(merger_seg_path, names=['merged'])
      print("File read: ", merger_seg_path)

  # This can only be called after preproc_data() is called
  def get_wv_model(self, force_build = True):
    if (not force_build) and os.access(save_wv_model_path, os.F_OK):
      self.wv_model = Word2Vec.load(save_wv_model_path)
      self.train_df['X'] = pd.read_csv(train_x_pad_path)
      self.train_df['Y'] = pd.read_csv(train_y_pad_path)
      self.test_df['X'] = pd.read_csv(test_x_pad_path)
      # The last line is always a NAN, remove it
      self.train_df = self.train_df.iloc[:][:-1]
      self.test_df = self.test_df.iloc[:][:-1]

      self.Y_max_len = get_max_len(self.train_df['Y'])
      self.X_max_len = max(get_max_len(self.train_df['X']), get_max_len(self.test_df['X']))
    else:
      #first, build upon all words include X, Y, train and test
      self.wv_model = Word2Vec(LineSentence(merger_seg_path),
                size=wv_embedding_dim,
                sg=1,
                workers=wv_worker_num,
                iter=wv_train_epochs,
                window=wv_ngram,
                min_count=wv_min_count)
      self.vocab = self.wv_model.wv.vocab
  
      # Split X Y value
      self.train_df['X'] = self.train_df[self.x_cols].apply(lambda x: ' '.join(x), axis=1)
      self.test_df['X'] = self.test_df[self.x_cols].apply(lambda x: ' '.join(x), axis=1)
      self.X_max_len = max(get_max_len(self.train_df['X']), get_max_len(self.test_df['X']))
      
      # Using generated vacab to retrain wv
      self.train_df['X'] = self.train_df['X'].apply(lambda x: pad_proc(x, self.X_max_len, self.vocab))
      self.test_df['X'] = self.test_df['X'].apply(lambda x: pad_proc(x, self.X_max_len, self.vocab))
      # Get get_max_len again, because something is added
      self.X_max_len = max(get_max_len(self.train_df['X']), get_max_len(self.test_df['X']))
  
      self.Y_max_len = get_max_len(self.train_df[self.y_cols[0]])
      self.train_df['Y'] =self.train_df[self.y_cols[0]].apply(lambda x: pad_proc(x, self.Y_max_len, self.vocab))
      # Get get_max_len again, because something is added
      self.Y_max_len = get_max_len(self.train_df['Y'])
      
      # Save splitted X Y
      self.train_df['X'].to_csv(train_x_pad_path, index=None, header=False)
      self.train_df['Y'].to_csv(train_y_pad_path, index=None, header=False)
      self.test_df['X'].to_csv(test_x_pad_path, index=None, header=False)
  
      # Update the wv again
      print('start retrain w2v model ...')
      self.wv_model.build_vocab(LineSentence(train_x_pad_path), update=True)
      self.wv_model.train(LineSentence(train_x_pad_path), epochs=wv_update_epochs, total_examples=self.wv_model.corpus_count)
      print('1/3')
      self.wv_model.build_vocab(LineSentence(train_y_pad_path), update=True)
      self.wv_model.train(LineSentence(train_y_pad_path), epochs=wv_update_epochs, total_examples=self.wv_model.corpus_count)
      print('2/3')
      self.wv_model.build_vocab(LineSentence(test_x_pad_path), update=True)
      self.wv_model.train(LineSentence(test_x_pad_path), epochs=wv_update_epochs, total_examples=self.wv_model.corpus_count)
      print('retrain w2v model done')
      
      # 保存词向量模型
      self.wv_model.save(save_wv_model_path)
      print('finish retrain w2v model')
      print('final w2v_model has vocabulary of ', len(self.wv_model.wv.vocab))

    # 保存字典
    if force_build or not os.access(vocab_path, os.F_OK):
      self.vocab = {word: index for index, word in enumerate(self.wv_model.wv.index2word)}
      self.reverse_vocab = {index: word for index, word in enumerate(self.wv_model.wv.index2word)}
      save_dict(vocab_path, self.vocab)
      save_dict(reverse_vocab_path, self.reverse_vocab)
    else:
      self.vocab = load_dict(vocab_path, (False, True))
      self.reverse_vocab = load_dict(reverse_vocab_path, (True, False))

    # 13. 保存词向量矩阵
    self.embedding_matrix = self.wv_model.wv.vectors
    if force_build or not os.access(embedding_matrix_path, os.F_OK):
      np.save(embedding_matrix_path, self.embedding_matrix)
      
    # At last, we get the id for special tokens
    self.token_id_start = self.vocab[self.WORD_START]
    self.token_id_stop = self.vocab[self.WORD_STOP]
    self.token_id_unk = self.vocab[self.WORD_UNK]
    self.token_id_pad = self.vocab[self.WORD_PAD]
    
    # Load the word list
    self.vocab_word_list = sorted(list(self.vocab.keys()))
    self.vocab_max_id = len(self.vocab) # id starts from 0
    
    # train_y_max_len will be removed later
    self.train_y_max_len = self.Y_max_len
  
  # This is time-consuming, so do not call it in initialization
  def get_data_in_id(self, force_build = True):
    if force_build or not os.access(train_x_path, os.F_OK):
      # 14. 数据集转换 将词转换成索引  [<START> 方向机 重 ...] -> [32800, 403, 986, 246, 231
      self.train_ids_x = self.train_df['X'].apply(lambda x: transform_data(x, self.vocab))
      self.train_ids_y = self.train_df['Y'].apply(lambda x: transform_data(x, self.vocab))
      self.test_ids_x = self.test_df['X'].apply(lambda x: transform_data(x, self.vocab))
    
      # 15. 数据转换成numpy数组
      # 将索引列表转换成矩阵 [32800, 403, 986, 246, 231] --> array([[32800,   403,   986 ]]
      self.train_X = np.array(self.train_ids_x.tolist())
      self.train_Y = np.array(self.train_ids_y.tolist())
      self.test_X = np.array(self.test_ids_x.tolist())
  
      # 保存数据
      np.save(train_x_path, self.train_X)
      np.save(train_y_path, self.train_Y)
      np.save(test_x_path, self.test_X)
    else:
      self.train_X = np.load(train_x_path + '.npy')
      self.train_Y = np.load(train_y_path + '.npy')
      self.test_X = np.load(test_x_path + '.npy')
      
      self.train_ids_x = np.array(self.train_X)
      self.train_ids_y = np.array(self.train_Y)
      self.test_ids_x = np.array(self.test_X)

    
  def tokens_to_texts(self, tokens):
    txt = []
    for t in tokens:
      txt.append(self.reverse_vocab[t])
    return ' '.join(txt)

  def texts_to_tokens(self, doc, sperator=' '):
    txts = doc.split(sperator = sperator)
    tokens = []
    for t in txts:
      tokens.append(self.word_to_id(t))
    return tokens
    
  def word_to_id(self, word):
    if word in self.vocab_word_list:
      return self.vocab[word]
    else:
      return self.token_id_unk
  
  def id_to_word(self, id):
    if id < self.vocab_max_id:
      return self.reverse_vocab[id]
    else:
      return self.WORD_UNK
      
  @property
  def vocab_size(self):
    return self.vocab_max_id
   

class AutoCarDataSet(BaseDataSet):
  def __init__(self):
    super(AutoCarDataSet, self).__init__(train_csv=train_data_path, test_csv=test_data_path, x_cols=['Question', 'Dialogue'], y_cols=['Report'])
    pass

