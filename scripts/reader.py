import pickle
import random
from scripts.model import convertToLSTMTree
from scripts.preprocess import convertDataSet,read_file
def read_data(DEBUG=False):
    training_pairs,test_pairs=[],[]
    with open('temp/data_pairs.pkl','rb') as f:
        training_pairs_O=pickle.load(f)
    with open('temp/test_pairs.pkl','rb') as f:
        test_pairs_O=pickle.load(f)
    with open('temp/word_dict.pkl','rb') as f:
        word_dict=pickle.load(f)
    if(DEBUG):
        (T1,T2),label=training_pairs_O[0]; 
        T1.print("")
    training_pairs=convertDataSet(training_pairs_O,word_dict,'training')
    test_pairs=convertDataSet(test_pairs_O,word_dict,'test')
    random.shuffle(training_pairs)
    random.shuffle(test_pairs)
    return training_pairs,test_pairs,word_dict

def read_single_file(file,word_dict):
    tree=read_file(file)
    return convertToLSTMTree(tree,word_dict)
