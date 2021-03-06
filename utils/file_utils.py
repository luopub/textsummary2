# -*- coding:utf-8 -*-
# Created by LuoJie at 11/27/19
import os

from utils.config import save_result_dir
import time


def save_vocab(file_path, data):
    with open(file_path) as f:
        for i in data:
            f.write(i)


def get_result_filename(batch_size, epochs, max_length_inp, embedding_dim, commit=''):
    """
    获取时间
    :return:
    """
    now_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    filename = now_time + '_batch_size_{}_epochs_{}_max_length_inp_{}_embedding_dim_{}{}.csv'.format(batch_size, epochs,
                                                                                                   max_length_inp,
                                                                                                   embedding_dim,
                                                                                                   commit)
    result_save_path = os.path.join(save_result_dir, filename)
    return result_save_path


def save_dict(save_path, dict_data):
    """
    保存字典
    :param save_path: 保存路径
    :param dict_data: 字典路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("{}\t{}\n".format(k, v))

#is_int is a tupple indicates which item is integer
def load_dict(file_path, is_int):
    """
    读取字典
    :param file_path: 文件路径
    :return: 返回读取后的字典
    """
    # return dict((line.strip().split("\t")[0], idx)
    #            for idx, line in enumerate(open(file_path, "r", encoding='utf-8').readlines()))
    dict = {}
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            kv = line.strip().split("\t")
            k = int(kv[0]) if is_int[0] else kv[0]
            v = int(kv[1]) if is_int[1] else kv[1]
            dict[k] = v
    return dict
