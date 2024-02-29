import pickle
import sys
import os

TRAIN_SIZE=1000
TEST_SIZE=100

def myMakedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def dump_cpp(path,text):
    with open(path,'w') as file:
        file.write(text)

def dump_info(root_path,programs,dict,map):
    myMakedir(root_path+'oj_clone_programs/')
    for i in dict:
        path=root_path+'oj_clone_programs/'+str(i)+'.cpp'
        dump_cpp(path,programs[1][i])
    file_map=open(root_path+'oj_clone_mapping.pkl','wb')
    pickle.dump(map,file_map)

if __name__=='__main__':
    vis_training,vis_test={},{}
    map_training,map_test=[],[]
    file_ids=open('data_set/OJ_Clone/oj_clone_ids.pkl','rb')
    ids=pickle.load(file_ids)
    count=0

    for i in range(len(ids)):
        if(ids.label[i]==1):
            count=count+1
            if(count<=TRAIN_SIZE):
                vis_training[ids.id1[i]]=1
                vis_training[ids.id2[i]]=1
                map_training.append(((ids.id1[i],ids.id2[i]),1))
            elif(count<=TRAIN_SIZE+TEST_SIZE):
                vis_test[ids.id1[i]]=1
                vis_test[ids.id2[i]]=1 
                map_test.append(((ids.id1[i],ids.id2[i]),1))
            else:
                break
    print('same pairs:',count)
    count=0
    for i in range(len(ids)):
        if(ids.label[i]==0):
            count=count+1
            if(count<=TRAIN_SIZE):
                vis_training[ids.id1[i]]=1
                vis_training[ids.id2[i]]=1
                map_training.append(((ids.id1[i],ids.id2[i]),-1))
            elif(count<=TRAIN_SIZE+TEST_SIZE):
                vis_test[ids.id1[i]]=1
                vis_test[ids.id2[i]]=1 
                map_test.append(((ids.id1[i],ids.id2[i]),-1))
            else:
                break
    print('different pairs:',count)
    myMakedir('data')
    myMakedir('data/training')
    myMakedir('data/test')
    file_programs=open('data_set/OJ_Clone/oj_clone_programs.pkl','rb')
    programs=pickle.load(file_programs)
    dump_info('data/training/',programs,vis_training,map_training)
    dump_info('data/test/',programs,vis_test,map_test)
