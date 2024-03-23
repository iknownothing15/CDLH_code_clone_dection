import torch
from tqdm import tqdm
from scripts.reader import read_single_file

def evaluate(test_pairs,word_dict,file='model.pt'):
    model=torch.load(file)
    correct = 0
    total = 0
    TP,TN,FP,FN=0,0,0,0
    with torch.no_grad():  # 在评估模式下，我们不需要计算梯度
        for pair, label in tqdm(test_pairs,desc="testing"):
            output1, _ = model(pair[0])
            output2, _ = model(pair[1])
            distance = torch.pairwise_distance(output1.view(1, -1), output2.view(1, -1), p=1)
            predict = 1 if distance < 0.5 else -1
            if predict == label:
                correct += 1
            if predict==1 and label==1:
                TP+=1
            if predict==-1 and label==-1:
                TN+=1
            if predict==1 and label==-1:
                FP+=1
            if predict==-1 and label==1:
                FN+=1
            total += 1
    # print(TP,TN,FP)
    Accuracy=correct / total
    Precision=TP/(TP+FP)
    Recall=TP/(TP+FN)
    F1=2*Precision*Recall/(Precision+Recall)
    return Accuracy,Precision,Recall,F1

def evaluate_single_pair(pair):
    model=torch.load('model.pt')
    with torch.no_grad():
        output1, _ = model(pair[0])
        output2, _ = model(pair[1])
        distance = torch.pairwise_distance(output1.view(1, -1), output2.view(1, -1), p=1)
        possbility_tensor = 1-distance
        possbility = possbility_tensor.item()
        return possbility

def evaluate_single(file1,file2,word_dict):
    LSTM_Tree1=read_single_file(file1,word_dict)
    LSTM_Tree2=read_single_file(file2,word_dict)
    possibility=evaluate_single_pair((LSTM_Tree1,LSTM_Tree2))
    return possibility

def get_evaluation_result(test_pairs,word_dict):
    Accuracy,Precision,Recall,F1=evaluate(test_pairs,word_dict)
    result ='Accuracy: %f\nPrecision: %f\nRecall: %f\nF1: %f\n' % (Accuracy,Precision,Recall,F1)
    return result