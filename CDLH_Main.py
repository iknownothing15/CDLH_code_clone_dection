from scripts.reader import read_data,read_single_file
from scripts.preprocess import convertDataSet
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from scripts.model import ChildSumTreeLSTM
import psutil
import os
EMBEDDING_DIM=32
HIDDEN_DIM=16

def train(training_pairs,word_dict,EPOCH=20):
    # training_pairs=convertDataSet(training_pairs_O,word_dict,'training')
    model=ChildSumTreeLSTM(EMBEDDING_DIM,HIDDEN_DIM,len(word_dict)+1)
    # model=torch.load('model.pt')
    loss_function=nn.HingeEmbeddingLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.1)
    for epoch in range(1,1+EPOCH):
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
            if i % 400==0:
                print('running loss of E%d-P%d: %f' % (epoch, i, running_loss / 400))
                running_loss = 0
        print('epoch %d: finish to train different codes' % epoch)
        print('average loss of epoch %d: %f' % (epoch, epoch_loss / len(training_pairs)))

    torch.save(model,'model.pt')
    

def evaluate(test_pairs,word_dict):
    model=torch.load('model.pt')
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下，我们不需要计算梯度
        for pair, label in tqdm(test_pairs,desc="testing"):
            output1, _ = model(pair[0])
            output2, _ = model(pair[1])
            distance = torch.pairwise_distance(output1.view(1, -1), output2.view(1, -1), p=1)
            predict = 1 if distance < 0.5 else -1
            if predict == label:
                correct += 1
            total += 1
    return 100 * correct / total



def evaluate_single_pair(pair):
    model=torch.load('model.pt')
    with torch.no_grad():
        output1, _ = model(pair[0])
        output2, _ = model(pair[1])
        distance = torch.pairwise_distance(output1.view(1, -1), output2.view(1, -1), p=1)
        possbility_tensor = 1-distance
        possbility = possbility_tensor.item()
        return possbility

def check(pairSample):
    pair,label=pairSample
    print('--------Cutting Line----------')
    pair[0].print('')
    print('--------Cutting Line----------')
    pair[1].print('')
    print('--------Cutting Line----------')

import psutil
import os

def resource_calc(test_pairs,word_dict):
    # 获取当前进程
    process = psutil.Process(os.getpid())

    # 记录开始时的资源使用情况
    start_resources = process.memory_info()
    start_cpu_time = process.cpu_times()

    # 执行函数
    evaluate(test_pairs,word_dict)

    # 记录结束时的资源使用情况
    end_resources = process.memory_info()
    end_cpu_time = process.cpu_times()

    # 计算并打印资源使用情况
    user_time = end_cpu_time.user - start_cpu_time.user
    system_time = end_cpu_time.system - start_cpu_time.system
    memory_usage = end_resources.rss - start_resources.rss

    print("User time: {:.2f} ms".format(user_time * 1000))
    print("System time: {:.2f} ms".format(system_time * 1000))
    print("Memory usage: {:.2f} KB".format(memory_usage / 1024))


def evaluate_single(file1,file2,word_dict):
    LSTM_Tree1=read_single_file(file1,word_dict)
    LSTM_Tree2=read_single_file(file2,word_dict)
    possibility=evaluate_single_pair((LSTM_Tree1,LSTM_Tree2))
    print(f"{possibility:.2f}")

def main():
    # init_ast()
    training_pairs_O,test_pairs_O,word_dict=read_data(DEBUG=False)
    # training_pairs=convertDataSet(training_pairs_O,word_dict,'training')
    test_pairs=convertDataSet(test_pairs_O,word_dict,'test')
    # train(training_pairs,word_dict)
    # evaluate(test_pairs,word_dict)
    resource_calc(test_pairs,word_dict)



if __name__ == "__main__":
    main()