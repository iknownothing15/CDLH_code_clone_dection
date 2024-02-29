import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import random
from scripts.preposs import read_data,init

EMBEDDING_DIM=32
HIDDEN_DIM=16

torch.random.manual_seed(1)
random.seed(114514)

def prepare_sequence(sentence,word_dict):
    idxs=[word_dict[word] for word in sentence]
    return torch.tensor(idxs)

class MyModule(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,target_size):
        super(MyModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_scores = functional.log_softmax(lstm_out.view(len(sentence), -1), dim=1)
        return tag_scores
    
def train(training_data,word_dict,EPOCH=20):
    model=MyModule(EMBEDDING_DIM,HIDDEN_DIM,len(word_dict),HIDDEN_DIM)
    # model=torch.load('model.pt')
    loss_function=nn.HingeEmbeddingLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.1)
    for epoch in range(EPOCH):
        i = 0
        epoch_loss = 0
        running_loss = 0
        # 遍历训练数据中的每一对句子和对应的标签
        for pair, label in training_data:
            # 清零模型的梯度
            model.zero_grad()
            # 初始化模型的隐状态
            model.hidden = model.init_hidden()
            sentence_1, sentence_2 = pair
            # 将句子转换为张量
            sentence_in_1 = prepare_sequence(sentence_1, word_dict)
            sentence_in_2 = prepare_sequence(sentence_2, word_dict)
            # 计算两个句子的最小长度
            min_size = min(len(sentence_1), len(sentence_2))
            index = - min_size

            # 前向传播，得到标签分数
            tag_scores_1 = model(sentence_in_1)[index:]
            tag_scores_2 = model(sentence_in_2)[index:]
            # 计算两个标签分数的距离
            distance = functional.pairwise_distance(tag_scores_1.view(1, -1), tag_scores_2.view(1, -1), p=1)
            # 计算损失
            loss = loss_function(distance, torch.tensor([label]))
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            epoch_loss += loss.item()
            i += 1
            if i % 500 == 499:
                print(epoch, i + 1, "running loss: ", running_loss / 499)
                running_loss = 0
        print('epoch %d: finish to train different codes' % epoch)
        print('average loss of epoch %d: %f' % (epoch, epoch_loss / len(training_data)))
        print('save model to file')
        # 保存模型
        torch.save(model, 'model.pt')
        epoch_loss = 0

def evaluate(test_pairs,word_dict):
    model=torch.load('model.pt')
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下，我们不需要计算梯度
        for pair, label in test_pairs:
            model.hidden=model.init_hidden()
            sentence_1, sentence_2 = pair
            sentence_in_1 = prepare_sequence(sentence_1, word_dict)
            sentence_in_2 = prepare_sequence(sentence_2, word_dict)
            min_size = min(len(sentence_1), len(sentence_2))
            index = - min_size
            tag_scores_1 = model(sentence_in_1)[index:]
            tag_scores_2 = model(sentence_in_2)[index:]
            distance = functional.pairwise_distance(tag_scores_1.view(1, -1), tag_scores_2.view(1, -1), p=1)
            predicted = 1.0 if distance.item() < 0.5 else -1.0  # 这里我们假设距离小于0.5的两个句子是相似的
            # print('id=%d distance=%f' % (id, distance.item()))
            if predicted == label:
                correct += 1
            total += 1
    print('Accuracy of the model on the test pairs: {}%'.format(100 * correct / total))

if __name__ == '__main__':
    # init()
    training_pairs,test_pairs,word_dict=read_data()
    # train(training_pairs,word_dict)
    evaluate(test_pairs+training_pairs,word_dict)