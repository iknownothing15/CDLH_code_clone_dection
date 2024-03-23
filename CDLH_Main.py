from scripts.reader import read_data
from scripts.preprocess import convertDataSet,init_ast
from train import mix_training
from evaluate import get_evaluation_result

def main():
    # init_ast()
    training_pairs_O,test_pairs_O,word_dict=read_data(DEBUG=False)
    training_pairs=convertDataSet(training_pairs_O,word_dict,'training')
    test_pairs=convertDataSet(test_pairs_O,word_dict,'test')
    mix_training(training_pairs,test_pairs,word_dict,)
    
if __name__ == "__main__":
    main()