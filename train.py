import time
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from evaluate import evaluate
from scripts.model import ChildSumTreeLSTM
from settings.consts import EMBEDDING_DIM,HIDDEN_DIM

def train(training_pairs,word_dict,EPOCHL,EPOCHR):
    if EPOCHL==1:
        model=ChildSumTreeLSTM(EMBEDDING_DIM,HIDDEN_DIM,len(word_dict)+1)
    else :
        model=torch.load('model.pt')
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
    torch.save(model,'model.pt')
    torch.save(model,'model/trained_model_epoch%d.pt' % epoch)

def mix_training(training_pairs,test_pairs,word_dict,EPOCHL,EPOCHR):
    # 初始化日志
    log = pd.DataFrame(columns=['Epoch','Accuracy','Precision','Recall','F1','Elapsed time'])

    for epoch in range(EPOCHL,EPOCHR+1):
        # 记录开始时的资源使用情况和时间
        start_time = time.time()

        train(training_pairs,word_dict,epoch,epoch)
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