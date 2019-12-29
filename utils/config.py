# -*- coding:utf-8 -*-
# Created by LuoJie at 11/16/19
import os
import pathlib
import sys

# 获取项目根目录
# project_root = pathlib.Path(os.path.abspath(__file__)).parent.parent
# project_root = pathlib.Path('/home/aistudio/work/project')
project_root = pathlib.Path(r'/gdrive/My Drive/python-code/文本摘要-01')
sys.path.append(str(project_root))

# 预处理数据 构建数据集
is_build_dataset = True

# 训练数据路径
train_data_path = os.path.join(project_root, 'data', 'AutoMaster_TrainSet.csv')
# 测试数据路径
test_data_path = os.path.join(project_root, 'data', 'AutoMaster_TestSet.csv')
# 停用词路径
# stop_word_path = os.path.join(project_root, 'data', 'stopwords/哈工大停用词表.txt')
stop_word_path = os.path.join(project_root, 'data', 'stopwords/stopwords.txt')

# 自定义切词表
user_dict = os.path.join(project_root, 'data', 'user_dict.txt')

# 0. 预处理
# 预处理后的训练数据
train_seg_path = os.path.join(project_root, 'data', 'train_seg_data.csv')
# 预处理后的测试数据
test_seg_path = os.path.join(project_root, 'data', 'test_seg_data.csv')
# 合并训练集测试集数据
merger_seg_path = os.path.join(project_root, 'data', 'merged_train_test_seg_data.csv')

# 1. 数据标签分离
train_x_seg_path = os.path.join(project_root, 'data', 'train_X_seg_data.csv')
train_y_seg_path = os.path.join(project_root, 'data', 'train_Y_seg_data.csv')
test_x_seg_path = os.path.join(project_root, 'data', 'test_X_seg_data.csv')

# 2. pad oov处理后的数据
train_x_pad_path = os.path.join(project_root, 'data', 'train_X_pad_data.csv')
train_y_pad_path = os.path.join(project_root, 'data', 'train_Y_pad_data.csv')
test_x_pad_path = os.path.join(project_root, 'data', 'test_X_pad_data.csv')

# 3. numpy 转换后的数据
train_x_path = os.path.join(project_root, 'data', 'train_X')
train_y_path = os.path.join(project_root, 'data', 'train_Y')
test_x_path = os.path.join(project_root, 'data', 'test_X')


# 以下是word vector相关的参数 ====================

# 词向量路径
save_wv_model_path = os.path.join(project_root, 'data', 'wv', 'word2vec.model')
# 词向量矩阵保存路径
embedding_matrix_path = os.path.join(project_root, 'data', 'wv', 'embedding_matrix')
# 字典路径
vocab_path = os.path.join(project_root, 'data', 'wv', 'vocab.txt')
reverse_vocab_path = os.path.join(project_root, 'data', 'wv', 'reverse_vocab.txt')

# 词向量维度
wv_embedding_dim = 300

# 词向量训练轮数
wv_train_epochs = 10
wv_update_epochs = 2

# Number of working process used in word2vec training
wv_worker_num = 2

# Number of neighbours used in skip-gram
wv_ngram = 5

# Minimum count for effective word when embedding
wv_min_count = 5

# 以下
# 模型保存文件夹
# checkpoints 保存路径
checkpoint_dir_tf = os.path.join(project_root, 'checkpoints', 'tf')
checkpoint_dir_paddle = os.path.join(project_root, 'checkpoints', 'paddle')

# checkpoint_dir = os.path.join(project_root, 'data', 'checkpoints', 'training_checkpoints_mask_loss_dim500_seq')

# checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

# 结果保存文件夹
save_result_dir = os.path.join(project_root, 'result')
