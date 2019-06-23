import sys
sys.path.append('/home/student401/MakiFlow')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
from makiflow.rnn_models.seq_predictor import SequencePredictor
from makiflow.rnn_layers import GRULayer, EmbeddingLayer
from brown import get_sentences_with_word2idx_limit_vocab
import tensorflow as tf
from copy import copy

session = tf.Session()

