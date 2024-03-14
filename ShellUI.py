import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from scripts.preprocess import convertDataSet
from scripts.reader import read_data,read_single_file,read_word_dict
from CDLH_Main import evaluate_single_pair,EMBEDDING_DIM,HIDDEN_DIM
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from scripts.model import ChildSumTreeLSTM
from tqdm import tqdm

word_dict = None
training_pairs_O = None
test_pairs_O = None
progress_bar_train = None
progress_bar_test =None

def UI_launch_evalvate(text1_content,text2_content):
    evaluating_window = tk.Toplevel(judge_code_window)
    evaluating_window.title("评估中")

    message = tk.Label(evaluating_window, text="评估中，请稍候...")
    message.pack()

    text1_content = text1_content.strip()
    text2_content = text2_content.strip()
    if text1_content == "" or text2_content == "":
        messagebox.showerror("错误", "输入不能为空")
        return
    if text1_content.find("#include") != -1 or text2_content.find("#include") != -1:
        messagebox.showerror("错误", "请不要包含include语句")
        return
    # print(text1_content)
    # print(text2_content)
    evaluating_window.update()
    with open("./data/inference/1.cpp", "w") as f:
        f.write(text1_content)
    with open("./data/inference/2.cpp", "w") as f:
        f.write(text2_content)
    global word_dict
    LSTM_Tree1 = read_single_file("./data/inference/1.cpp", word_dict)
    LSTM_Tree2 = read_single_file("./data/inference/2.cpp", word_dict)
    possibility=evaluate_single_pair((LSTM_Tree1,LSTM_Tree2))
    evaluating_window.destroy()
    messagebox.showinfo("结果", f"相似度为{possibility:.2f}")

def train(training_pairs,word_dict,EPOCH=20):
    global progress_bar_train
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

            if i % 10 ==0 :
                progress_bar_train['value'] = (i / len(training_pairs)) * 100
                start_menu.update_idletasks()
            # print(i)
        messagebox.showinfo("训练进度", 'epoch %d: 完成训练不同的代码' % epoch)
        messagebox.showinfo("训练进度", 'epoch %d的平均损失: %f' % (epoch, epoch_loss / len(training_pairs)))
    torch.save(model,'model.pt')


def train_launch():
    # training_pairs_O,test_pairs_O,word_dict=read_data(DEBUG=False)
    global training_pairs_O,word_dict
    training_pairs=convertDataSet(training_pairs_O,word_dict,'training')
    # test_pairs=convertDataSet(test_pairs_O,word_dict,'test')
    train_thread=threading.Thread(target=train,args=(training_pairs,word_dict,20))
    train_thread.start()

def init_show_train_model():
    global progress_bar_train
    intro_text="即将对模型进行训练，将使用temp\data_pairs.pkl中的数据"
    intro_function = tk.Label(train_model_window, text=intro_text)
    intro_function.grid(row=0, column=0, columnspan=3, sticky='ew')

    progress_text = tk.Label(train_model_window, text="训练进度：")
    progress_text.grid(row=1, column=0, sticky='ew')

    progress_bar_train = ttk.Progressbar(train_model_window, orient="horizontal", length=200, mode="determinate")
    progress_bar_train.grid(row=1, column=1, columnspan=2, sticky='ew')

    start_train_button = tk.Button(train_model_window, text="开始训练", command=train_launch)
    start_train_button.grid(row=3, column=0, sticky='ew')

    exit_button = tk.Button(train_model_window, text="返回", command=show_menu)
    exit_button.grid(row=3, column=1, sticky='ew')

def show_train_model():
    start_menu.withdraw()
    train_model_window.deiconify()

def evaluate(test_pairs,word_dict):
    model=torch.load('model.pt')
    global progress_bar_test
    correct = 0
    total = 0
    i = 0
    with torch.no_grad():  # 在评估模式下，我们不需要计算梯度
        for pair, label in tqdm(test_pairs,desc="testing"):
            i += 1
            output1, _ = model(pair[0])
            output2, _ = model(pair[1])
            distance = torch.pairwise_distance(output1.view(1, -1), output2.view(1, -1), p=1)
            predict = 1 if distance < 0.5 else -1
            if predict == label:
                correct += 1
            total += 1
            progress_bar_test['value'] = (i / len(test_pairs)) * 100
            start_menu.update_idletasks()
    result = 100 * correct / total
    messagebox.showinfo("评估结果", f"准确率为{result:.2f}%")

def evaluate_launch():
    global test_pairs_O,word_dict
    test_pairs=convertDataSet(test_pairs_O,word_dict,'test')
    evaluate_thread=threading.Thread(target=evaluate,args=(test_pairs,word_dict))
    evaluate_thread.start()

def init_show_evaluate_model():
    global progress_bar_test
    intro_text="即将对模型进行评估，将使用temp\\test_pairs.pkl中的数据"
    intro_function = tk.Label(evaluate_model_window, text=intro_text)
    intro_function.grid(row=0, column=0, columnspan=3,sticky='ew')

    progress_text = tk.Label(evaluate_model_window, text="评估进度：")
    progress_text.grid(row=1, column=0,sticky='ew')

    progress_bar_test = ttk.Progressbar(evaluate_model_window, orient="horizontal", length=200, mode="determinate")
    progress_bar_test.grid(row=1, column=1, columnspan=2,sticky='ew')

    start_train_button = tk.Button(evaluate_model_window, text="开始评估", command=evaluate_launch)
    start_train_button.grid(row=3, column=0,sticky='ew')

    exit_button = tk.Button(evaluate_model_window, text="返回", command=show_menu)
    exit_button.grid(row=3, column=1,sticky='ew')

def show_evaluate_model():
    start_menu.withdraw()
    evaluate_model_window.deiconify()

def init_show_judge_code():
    intro_function = tk.Label(judge_code_window, text="您可以在以下输入框中输入您的代码，我们将会对您的代码进行评估。\n请注意不要包含include语句。")
    intro_function.grid(row=0, column=0, columnspan=2)

    label1 = tk.Label(judge_code_window, text="待检测代码1")
    label1.grid(row=1, column=0)

    label2 = tk.Label(judge_code_window, text="待检测代码2")
    label2.grid(row=1, column=1)

    frame1 = tk.Frame(judge_code_window)
    frame1.grid(row=2, column=0, sticky="nsew")

    text1 = tk.Text(frame1)
    text1.pack(side='left', fill='both', expand=True)

    scrollbar1 = tk.Scrollbar(frame1, command=text1.yview)
    scrollbar1.pack(side='right', fill='y')

    text1.config(yscrollcommand=scrollbar1.set)

    frame2 = tk.Frame(judge_code_window)
    frame2.grid(row=2, column=1, sticky="nsew")

    text2 = tk.Text(frame2)
    text2.pack(side='left', fill='both', expand=True)

    scrollbar2 = tk.Scrollbar(frame2, command=text2.yview)
    scrollbar2.pack(side='right', fill='y')

    text2.config(yscrollcommand=scrollbar2.set)

    button_launch = tk.Button(judge_code_window, text="开始检测", command=lambda: UI_launch_evalvate(text1.get("1.0", 'end-1c'), text2.get("1.0", 'end-1c')))
    button_launch.grid(row=3, column=0)

    button_exit =tk.Button(judge_code_window, text="返回", command=show_menu)
    button_exit.grid(row=3, column=1)

    judge_code_window.grid_rowconfigure(2, weight=1)
    judge_code_window.grid_columnconfigure(0, weight=1)
    judge_code_window.grid_columnconfigure(1, weight=1)

def show_judge_code():
    start_menu.withdraw()
    judge_code_window.deiconify()

def init_menu():

    train_button = tk.Button(start_menu, text="模型训练", command=show_train_model)
    train_button.pack(padx=20, pady=10, anchor='center')

    evaluate_button = tk.Button(start_menu, text="模型评估", command=show_evaluate_model)
    evaluate_button.pack(padx=20, pady=10, anchor='center')

    judge_button = tk.Button(start_menu, text="代码对判断", command=show_judge_code)
    judge_button.pack(padx=20, pady=10, anchor='center')

    exit_button = tk.Button(start_menu, text="退出", command=start_menu.quit)
    exit_button.pack(padx=20, pady=10, anchor='center')

def show_menu():
    train_model_window.withdraw()
    evaluate_model_window.withdraw()
    judge_code_window.withdraw()
    start_menu.deiconify()

training_pairs_O,test_pairs_O,word_dict=read_data(DEBUG=False)

start_menu = tk.Tk()
start_menu.title("代码相似度检测")
start_menu.geometry("300x200")

train_model_window = tk.Toplevel(start_menu)
train_model_window.withdraw()
train_model_window.title("模型训练")

evaluate_model_window = tk.Toplevel(start_menu)
evaluate_model_window.withdraw()

judge_code_window = tk.Toplevel(start_menu)
judge_code_window.withdraw()

init_menu()
init_show_judge_code()
init_show_evaluate_model()
init_show_train_model()
show_menu()

start_menu.mainloop()