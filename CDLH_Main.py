from scripts.preprocess import read_data_Ontime,init_ast,convert_pairs
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from scripts.model import ChildSumTreeLSTM,convertToLSTMTree
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
    # test_pairs=convertDataSet(test_pairs_O,word_dict,'test')
    model=torch.load('model.pt')
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下，我们不需要计算梯度
        for pair, label in test_pairs:
            output1, _ = model(pair[0])
            output2, _ = model(pair[1])
            distance = torch.pairwise_distance(output1.view(1, -1), output2.view(1, -1), p=1)
            predict = 1 if distance < 0.5 else -1
            if predict == label:
                correct += 1
            total+=1
    print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))
    return 100 * correct / total

def check(pairSample):
    pair,label=pairSample
    print('--------Cutting Line----------')
    pair[0].print('')
    print('--------Cutting Line----------')
    pair[1].print('')
    print('--------Cutting Line----------')


def main():
    # init_ast()
    # convert_pairs()
    training_pairs,test_pairs,word_dict=read_data_Ontime()
    # train(training_pairs,word_dict)
    evaluate(test_pairs+training_pairs,word_dict)

if __name__ == "__main__":
    main()