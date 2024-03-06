import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def convertToId(word,word_dict):
    if word in word_dict:
        return word_dict[word]
    else:
        return len(word_dict)

def convertToLSTMTree(tree,word_dict):
    if tree.num_children==0:
        return LSTMtree(convertToId(tree.attribute,word_dict))
    else:
        children=[]
        for child in tree.children:
            children.append(convertToLSTMTree(child,word_dict))
        return LSTMtree(convertToId(tree.attribute,word_dict),children)

class LSTMtree:
    def __init__(self, attribute, children=None):
        self.attribute = torch.tensor(attribute, dtype=torch.long)
        self.children = children if children is not None else []
        self.num_children = len(self.children)
        self.state = None
    def get_children_states(self):
        # 获取children的状态
        children_c = []
        children_h = []
        for child in self.children:
            if child.state is not None:
                c, h = child.state
                children_c.append(c)
                children_h.append(h)
        return children_c, children_h

class ChildSumTreeLSTM(nn.Module):
    # 初始化方法，定义了一些线性层和参数
    def __init__(self, in_dim, mem_dim,word_dict_size):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.embedding = nn.Embedding(word_dict_size, in_dim)
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    # 定义 node_forward 方法，用于计算每个节点的输出
    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h
    
    # 定义 forward 方法，用于前向传播
    def forward(self, tree):
        inputs=self.embedding(tree.attribute)
        for subtree in tree.children:
            self.forward(subtree)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        tree.state = self.node_forward(inputs, child_c, child_h)
        return tree.state
