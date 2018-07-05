# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import urllib.request
import re
import pickle
import time

import pandas as pd
import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

    Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = word_to_id
    return train_data, valid_data, test_data, vocabulary



def _print_download_progress(count, block_size, total_size):
    """
    Function used for printing the download progress.
    Used as a call-back function in maybe_download_and_extract().
    """
    # Percentage completion.
    pct_complete = float(count * block_size) / total_size

    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()



def load_data(data_path=None, dataset='PTB'):
    """
    Args:
    data_path: String, data path
    dataset: String, support 'PTB' or 'WIKI-103'
    
    Returns: 
    train_data: List of word index, training data
    valid_data: .....
    test_data: .....
    
    word_to_id: dict, word to word index
    id_to_word: dict, word index to word
    word_id: DataFrame version of word_to_id
    id_word: DataFrame version of id_to_word
    voc_size: int, voc_size
    """
    assert dataset == 'PTB' or dataset == 'WIKI-103'
    data_path = data_path + '/' + dataset
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if dataset == 'PTB':
        train_data, valid_data, test_data, word_to_id = ptb_raw_data(data_path) # tokens


        id_to_word = dict((v, k) for k, v in word_to_id.items())
        voc_size = len(id_to_word)
        
        id_to_word[voc_size]='<SS>' # Add start word token '<SS>'
        id_to_word[voc_size+1]='<EE>' # Add end word token '<EE>'
        word_to_id = dict((v, k) for k, v in id_to_word.items())
        voc_size = len(id_to_word)
        word_id = pd.DataFrame.from_dict(word_to_id, orient='index').sort_values(by=0, ascending=True)
        word_id.columns = ['id']
        id_word = pd.DataFrame.from_dict(id_to_word, orient='index')
        id_word.columns = ['word']
        
        train_dataDF = pd.DataFrame(train_data, columns=['id'])
        valid_dataDF = pd.DataFrame(valid_data, columns=['id'])
        test_dataDF = pd.DataFrame(test_data, columns=['id'])
    
    elif dataset == 'WIKI-103':
        urlDict = {"train": "http://www.stikbuf.top:8080/wikitext-103/wiki.train.tokens", 
        "valid": "http://www.stikbuf.top:8080/wikitext-103/wiki.valid.tokens", 
        "test": "http://www.stikbuf.top:8080/wikitext-103/wiki.test.tokens"}

        p = [re.findall('wiki\..*\.tokens', path) for path in os.listdir(data_path)]
        p = list(filter(lambda a: a != [], p))

        if len(p) != 3:
            for dataset in urlDict.keys():
                print('Downloading {0} set'.format(dataset))
                urllib.request.urlretrieve(url=urlDict[dataset],
                                       filename=os.path.join(data_path, urlDict[dataset].split('/')[-1]),
                                       reporthook=_print_download_progress)
                print('\n')
        else:
            print('Wikitext-103 dataset has been downloaded.')
        os.listdir(data_path)
    
        Py3 = sys.version_info[0] == 3

        train_path = os.path.join(data_path, "wiki.train.tokens")
        valid_path = os.path.join(data_path, "wiki.valid.tokens")
        test_path = os.path.join(data_path, "wiki.test.tokens")

        start = time.time()
        if not os.path.exists(os.path.join(data_path, 'dict.pkl')):
            print("Building vocabulary...")
            word_to_id = _build_vocab(train_path) # about 30s in my computer
            pickle.dump(word_to_id, open(os.path.join(data_path, 'dict.pkl'), "wb"))
            print('Building vocabulary cost {}s'.format(time.time()-start))
        else:
            print("Loading vocabulary...")
            word_to_id = pickle.load(open(os.path.join(data_path, 'dict.pkl'), "rb"))
            print('Loading vocabulary cost {}s'.format(time.time()-start))            
        
        start = time.time()
        if not os.path.exists(os.path.join(data_path, 'wiki.train.pkl')):
            print("Building training data...")
            train_data = _file_to_word_ids(train_path, word_to_id)
            pickle.dump(train_data, open(os.path.join(data_path, 'wiki.train.pkl'), "wb"))
        else:
            print("Loading training data...")
            train_data = pickle.load(open(os.path.join(data_path, 'wiki.train.pkl'), "rb"))

        if not os.path.exists(os.path.join(data_path, 'wiki.valid.pkl')):
            print("Building validation data...")
            valid_data = _file_to_word_ids(valid_path, word_to_id)
            pickle.dump(valid_data, open(os.path.join(data_path, 'wiki.valid.pkl'), "wb"))
        else:
            print("Loading validation data...")
            valid_data = pickle.load(open(os.path.join(data_path, 'wiki.valid.pkl'), "rb"))
            
        if not os.path.exists(os.path.join(data_path, 'wiki.test.pkl')):
            print("Building test data...")
            test_data = _file_to_word_ids(test_path, word_to_id)
            pickle.dump(test_data, open(os.path.join(data_path, 'wiki.test.pkl'), "wb"))
        else:
            print("Loading test data...")
            test_data = pickle.load(open(os.path.join(data_path, 'wiki.test.pkl'), "rb"))           
        print("Building/loading train-val-test data cost {}s".format(time.time() - start))  # about 35s in my computer
        
        id_to_word = dict((v, k) for k, v in word_to_id.items())
        voc_size = len(id_to_word)
        id_to_word[voc_size]='<SS>' # Add start word token '<SS>'
        id_to_word[voc_size+1]='<EE>' # Add end word token '<EE>'
        word_to_id = dict((v, k) for k, v in id_to_word.items())
        voc_size = len(id_to_word)
        
        start = time.time()
        if not os.path.exists(os.path.join(data_path, 'wiki.train.pd.pkl')):
            print("Building pandas training data...")
            train_dataDF = pd.DataFrame(train_data, columns=['id']) 
            train_dataDF.to_pickle(os.path.join(data_path, 'wiki.train.pd.pkl'))
        else:
            print("Reading pandas training data...")
            train_dataDF = pd.read_pickle(os.path.join(data_path, 'wiki.train.pd.pkl'))
              
        if not os.path.exists(os.path.join(data_path, 'wiki.valid.pd.pkl')):
            print("Building pandas validation data...")
            valid_dataDF = pd.DataFrame(valid_data, columns=['id']) 
            valid_dataDF.to_pickle(os.path.join(data_path, 'wiki.valid.pd.pkl'))
        else:
            print("Reading pandas validation data...")
            valid_dataDF = pd.read_pickle(os.path.join(data_path, 'wiki.valid.pd.pkl'))
                
        if not os.path.exists(os.path.join(data_path, 'wiki.test.pd.pkl')):
            print("Building pandas test data...")
            test_dataDF = pd.DataFrame(test_data, columns=['id']) 
            test_dataDF.to_pickle(os.path.join(data_path, 'wiki.test.pd.pkl'))
        else:
            print("Reading pandas test data...")
            test_dataDF = pd.read_pickle(os.path.join(data_path, 'wiki.test.pd.pkl'))    
        print("Building/loading pandas train-val-test data cost {}s".format(time.time() - start))
        
        start = time.time()
        print("Building pandas word_to_id dict...")
        word_id = pd.DataFrame.from_dict(word_to_id, orient='index').sort_values(by=0, ascending=True)
        word_id.columns = ['id']
        print("Building pandas id_to_word dict...")
        id_word = pd.DataFrame.from_dict(id_to_word, orient='index')
        id_word.columns = ['word']
        print("Building pandas dictionary cost {}s".format(time.time() - start))
        
        print('Done')
        
    return train_data, valid_data, test_data, \
                train_dataDF, valid_dataDF, test_dataDF, \
                word_to_id, id_to_word, \
                word_id, id_word, voc_size 

# def ptb_producer(raw_data, batch_size, num_steps, name=None):
#   """Iterate on the raw PTB data.

#   This chunks up raw_data into batches of examples and returns Tensors that
#   are drawn from these batches.

#   Args:
#     raw_data: one of the raw data outputs from ptb_raw_data.
#     batch_size: int, the batch size.
#     num_steps: int, the number of unrolls.
#     name: the name of this operation (optional).

#   Returns:
#     A pair of Tensors, each shaped [batch_size, num_steps]. The second element
#     of the tuple is the same data time-shifted to the right by one.

#   Raises:
#     tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
#   """
#   with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
#     raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

#     data_len = tf.size(raw_data)
#     batch_len = data_len // batch_size
#     data = tf.reshape(raw_data[0 : batch_size * batch_len],
#                       [batch_size, batch_len])

#     epoch_size = (batch_len - 1) // num_steps
#     assertion = tf.assert_positive(
#         epoch_size,
#         message="epoch_size == 0, decrease batch_size or num_steps")
#     with tf.control_dependencies([assertion]):
#       epoch_size = tf.identity(epoch_size, name="epoch_size")

#     i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
#     x = tf.strided_slice(data, [0, i * num_steps],
#                          [batch_size, (i + 1) * num_steps])
#     x.set_shape([batch_size, num_steps])
#     y = tf.strided_slice(data, [0, i * num_steps + 1],
#                          [batch_size, (i + 1) * num_steps + 1])
#     y.set_shape([batch_size, num_steps])
#     return x, y