from scripts.reader import read_data,read_single_file
from scripts.preprocess import convertDataSet,init_ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from scripts.model import ChildSumTreeLSTM
import psutil
import os
import pandas as pd
import time
EMBEDDING_DIM=32
HIDDEN_DIM=16

def train(training_pairs,word_dict,EPOCHL,EPOCHR,file=None):
    if file is None:
        model=ChildSumTreeLSTM(EMBEDDING_DIM,HIDDEN_DIM,len(word_dict)+1)
        file='model.pt'
    else :
        model=torch.load(file)
    loss_function=nn.HingeEmbeddingLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.01)
    for epoch in range(EPOCHL,EPOCHR+1):
        i = 1
        epoch_loss = 0
        running_loss = 0
        # 遍历训练数据中的每一对句子和对应的标签
        for pair, label in tqdm(training_pairs, desc='epoch %d' % epoch):
            i += 1 
            # 清除梯度
            model.zero_grad()
            # 计算预测的相似度
            output1, _ = model(pair[0])
            output2, _ = model(pair[1])
            distance = torch.pairwise_distance(output1.view(1, -1), output2.view(1, -1), p=1)
            #similarity = torch.cosine_similarity(output1, output2, dim=1)
            # 计算损失
            loss = loss_function(distance, torch.tensor([label]))
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 记录损失
            running_loss += loss.item()
            epoch_loss += loss.item()
            # print(i)
        print('epoch %d: finish to train different codes' % epoch)
        print('average loss of epoch %d: %f' % (epoch, epoch_loss / len(training_pairs)))
    torch.save(model,file)
    
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

def mix_training(training_pairs,test_pairs,word_dict,EPOCHS):
    # 初始化日志
    log = pd.DataFrame(columns=['Epoch','Accuracy','Precision','Recall','F1','Elapsed time'])

    for epoch in range(1,1+EPOCHS):
        # 记录开始时的资源使用情况和时间
        start_time = time.time()
        if epoch==1:
            train(training_pairs,word_dict,epoch,epoch)
        else:
            train(training_pairs,word_dict,epoch,epoch,'model.pt')
        Accuracy,Precision,Recall,F1=evaluate(test_pairs,word_dict)

        # 记录结束时的资源使用情况和时间
        end_time = time.time()

        # 计算资源使用情况和时间
        elapsed_time = end_time - start_time

        # 创建新的日志记录
        new_log = pd.DataFrame({
            'Epoch': [epoch],
            'Accuracy': [Accuracy],
            'Precision': [Precision],
            'Recall': [Recall],
            'F1': [F1],
            'Elapsed time': [elapsed_time]
        })

        # 将新的日志记录添加到现有的日志
        log = pd.concat([log, new_log], ignore_index=True)

        print(f"Accuracy of epoch {epoch}: {Accuracy*100:.2f}%")
        
        # 保存日志
        try:
            log.to_excel('./log/training_log.xlsx', index=False)
        except PermissionError:
            print("无法保存日志文件，因为文件已被其他程序打开。")
            log.to_csv('./log/training_log.csv', index=False)

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

def main():
    # init_ast()
    training_pairs_O,test_pairs_O,word_dict=read_data(DEBUG=False)
    training_pairs=convertDataSet(training_pairs_O,word_dict,'training')
    test_pairs=convertDataSet(test_pairs_O,word_dict,'test')
    mix_training(training_pairs,test_pairs,word_dict,20)
    # Accuracy,Precision,Recall,F1=evaluate(test_pairs,word_dict)



if __name__ == "__main__":
    main()