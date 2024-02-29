import os
import pickle
from tqdm import tqdm
import random
from pycparser import c_ast,c_parser,parse_file
from scripts.ASTVisitor import ASTVisitor,AST_Tree
from scripts.model import convertToLSTMTree

def read_file(filename):
    vistor=ASTVisitor()
    ast=parse_file(filename,use_cpp=True)
    tree=vistor.getAST(ast,None)
    return tree

def init_pairs(path,file_pairs):
    path = os.path.join(path, 'oj_clone_programs')
    pairs = []
    for filepair, label in tqdm(file_pairs, desc="Processing files"):
        treeA = read_file(os.path.join(path, str(filepair[0]) + '.cpp'))
        treeB = read_file(os.path.join(path, str(filepair[1]) + '.cpp'))
        pairs.append(((treeA, treeB), label))
    print('pairs:', len(pairs))
    return pairs

def convertDataSet(dataSet_O,word_dict,type):
    dataSet=[]
    sentence='Converting '+type+' DataSet'
    for pair,label in tqdm(dataSet_O,desc=sentence):
        pairxL=convertToLSTMTree(pair[0],word_dict)
        pairxR=convertToLSTMTree(pair[1],word_dict)
        dataSet.append(((pairxL,pairxR),label))
    return dataSet

def init_ast():
    # print("!!!DEBUG!!!")
    training_pairs,test_pairs=[],[]
    file_training=open('data/training/oj_clone_mapping.pkl','rb')
    file_test=open('data/test/oj_clone_mapping.pkl','rb')
    training_pairs=pickle.load(file_training)
    test_pairs=pickle.load(file_test)
    training_pairs=init_pairs('data/training',training_pairs)
    test_pairs=init_pairs('data/test',test_pairs)
    word_dict={}
    for pair in training_pairs+test_pairs:
        for tree in pair[0]:
            for attr in tree.get_all_attributes():
                if attr not in word_dict:
                    word_dict[attr]=len(word_dict)
    with open('temp/data_pairs_O.pkl','wb') as f:
        pickle.dump(training_pairs,f)
    with open('temp/test_pairs_O.pkl','wb') as f:
        pickle.dump(test_pairs,f)
    with open('temp/word_dict.pkl','wb') as f:
        pickle.dump(word_dict,f)

def convert_pairs():
    with open('temp/data_pairs_O.pkl','rb') as f:
        training_pairs_O=pickle.load(f)
    with open('temp/test_pairs_O.pkl','rb') as f:
        test_pairs_O=pickle.load(f)
    with open('temp/word_dict.pkl','rb') as f:
        word_dict=pickle.load(f)
    training_pairs=convertDataSet(training_pairs_O,word_dict,'training')
    test_pairs=convertDataSet(test_pairs_O,word_dict,'test')
    with open('temp/data_pairs.pkl','wb') as f:
        pickle.dump(training_pairs,f)
    with open('temp/test_pairs.pkl','wb') as f:
        pickle.dump(test_pairs,f)

def read_data_Ontime():
    training_pairs,test_pairs=[],[]
    with open('temp/data_pairs_O.pkl','rb') as f:
        training_pairs_O=pickle.load(f)
    with open('temp/test_pairs_O.pkl','rb') as f:
        test_pairs_O=pickle.load(f)
    with open('temp/word_dict.pkl','rb') as f:
        word_dict=pickle.load(f)
    training_pairs=convertDataSet(training_pairs_O,word_dict,'training')
    test_pairs=convertDataSet(test_pairs_O,word_dict,'test')
    random.shuffle(training_pairs)
    random.shuffle(test_pairs)
    return training_pairs,test_pairs,word_dict

def read_data():
    with open('temp/data_pairs.pkl','rb') as f:
        training_pairs=pickle.load(f)
    with open('temp/test_pairs.pkl','rb') as f:
        test_pairs=pickle.load(f)
    with open('temp/word_dict.pkl','rb') as f:
        word_dict=pickle.load(f)
    random.shuffle(training_pairs)
    random.shuffle(test_pairs)
    return training_pairs,test_pairs,word_dict