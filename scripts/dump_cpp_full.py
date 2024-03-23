import random
import pickle
import sys
import os
TRAIN_SIZE_S=500
TRAIN_SIZE_D=1500
TEST_SIZE_S=100
TEST_SIZE_D=100
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

def init_pairs(ids):
    mapS,mapD=[],[]
    for i in range(len(ids)):
        if(ids.label[i]==1):
            mapS.append(((ids.id1[i],ids.id2[i]),+1))
        elif(ids.label[i]==0):
            mapD.append(((ids.id1[i],ids.id2[i]),-1))
    return mapS,mapD

def dump_data_set(numS,numD,mapS,mapD):
    vis={}
    map=[]
    random.shuffle(mapS)
    random.shuffle(mapD)
    for i in range(numS):
        vis[mapS[i][0][0]]=1
        vis[mapS[i][0][1]]=1
        map.append(mapS[i])
    for i in range(numD):
        vis[mapD[i][0][0]]=1
        vis[mapD[i][0][1]]=1
        map.append(mapD[i])
    return vis,map

if __name__=='__main__':
    vis_training,vis_test={},{}
    map_training,map_test=[],[]
    # init
    file_ids=open('data_set/OJ_Clone/oj_clone_ids.pkl','rb')
    ids=pickle.load(file_ids)
    file_programs=open('data_set/OJ_Clone/oj_clone_programs.pkl','rb')
    programs=pickle.load(file_programs)
    # read
    mapS,mapD=init_pairs(ids)
    print(len(mapS),len(mapD))
    if(TRAIN_SIZE_S+TEST_SIZE_S>len(mapS)):
        print('Error: There are not enough positive samples')
        sys.exit()
    if(TRAIN_SIZE_D+TEST_SIZE_D>len(mapD)):
        print('Error: There are not enough negative samples')
        sys.exit()
    vis_training,map_training=dump_data_set(TRAIN_SIZE_S,TRAIN_SIZE_D,mapS,mapD)
    vis_test,map_test=dump_data_set(TEST_SIZE_S,TEST_SIZE_D,mapS,mapD)
    # label
    dump_info('data/training/',programs,vis_training,map_training)
    dump_info('data/test/',programs,vis_test,map_test)
    # dump