import copy
import re
import os


#gets data from babl task, structures them into memories, queries, and answers, returns 2 tuples of train, test data
def task_eval(task_num, task_dir="./tasks_1-20_v1-2/en"):
    
    task_files = sorted(os.listdir(task_dir))
    if not task_files[0].endswith('.txt'):  #there's a sneaky .ipynb file that was getting into this list, at beginning
        task_files = task_files[1:] 
        
    test_file = [f for f in task_files if f'qa{task_num}_' in f and 'test' in f][0]
    train_file = [f for f in task_files if f'qa{task_num}_' in f and 'train' in f][0]
    
    
    
    
    test_mems, test_queries, test_answers = structure_tasks(task_dir, test_file)
    train_mems, train_queries, train_answers = structure_tasks(task_dir, train_file)
    return (train_mems, train_queries, train_answers), (test_mems, test_queries, test_answers)



#return a tuple of memory (fact) sentences, query sentences, and answers, for every QA set in data
def structure_tasks(task_dir, filename):

   
    memories, queries, answers = [], [], []
    mems_i = []
    last = 0
    
    supporting = []

    for j, line in enumerate(open(task_dir + '/' + filename, 'r').readlines()):
        
        supp = []
        line = line.split()
        n = int(line[0])

        supp = [i for i in line if i.isnumeric()]
        supporting.append(supp)
        if line[-1].isnumeric():
            
            
            q_end = [i for i, w in enumerate(line) if w.endswith('?')]
            queries.append([line[1:q_end]])   #not keeping the strong supervision of the supporting memory indexes
            
            
            answers.append(line[q_end + 1])  #dont do lists this iteration...
           
           
            add_mems = copy.copy(mems_i)[::-1] #reverse as in 4.1 of article
            memories.append(add_mems)
            
            #print(len(answers), len(queries), len(memories))

        elif n >= last:  #should never be ==
            last = n
            mems_i.append(line[1:])

        elif n < last:
            last = n
            mems_i = [line[1:]]
            
   


    return memories, queries, answers



#NOT USED (yet)
def get_pre_trained(filename):

    pre_trained_embeddings = {}
    for line in open(filename, 'r').readlines():
        line = line.split()
        pre_trained_embeddings[line[0]] = np.array(line[1:])

    return pre_trained_embeddings
