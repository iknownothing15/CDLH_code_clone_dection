import pickle
import random
from scripts.model import convertToLSTMTree
from scripts.preprocess import convertDataSet,read_file
def read_data(DEBUG=False):
    training_pairs,test_pairs=[],[]
    with open('temp/data_pairs.pkl','rb') as f:
        training_pairs=pickle.load(f)
    with open('temp/test_pairs.pkl','rb') as f:
        test_pairs=pickle.load(f)
    with open('temp/word_dict.pkl','rb') as f:
        word_dict=pickle.load(f)
    random.shuffle(training_pairs)
    random.shuffle(test_pairs)
    return training_pairs,test_pairs,word_dict


def read_single_file(file,word_dict):
    tree=read_file(file)
    return convertToLSTMTree(tree,word_dict)
