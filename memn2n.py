

  
  
  
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
            
            
            q_end = [i for (i, w) in enumerate(line) if w.endswith('?')][0]
            
            queries.append([line[1:q_end + 1]])   #not keeping the strong supervision of the supporting memory indexes
            
            
            answers.append([line[-2]])  #dont do lists this iteration...
           
           
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




def task_eval(task_num, task_dir="./tasks_1-20_v1-2/en"):
    
    
    task_files = sorted(os.listdir(task_dir))
    if not task_files[0].endswith('.txt'):  #there's a sneaky .ipynb file that was getting into this list, at beginning
        task_files = task_files[1:] 
        
    test_file = [f for f in task_files if f'qa{task_num}_' in f and 'test' in f][0]
    train_file = [f for f in task_files if f'qa{task_num}_' in f and 'train' in f][0]
    
    test_mems, test_queries, test_answers = structure_tasks(task_dir, test_file)
    train_mems, train_queries, train_answers = structure_tasks(task_dir, train_file)
    return (train_mems, train_queries, train_answers), (test_mems, test_queries, test_answers)




def get_vocab(mems, queries, ans):
    vocab = set(v_w.strip(string.punctuation).lower() for m in mems for v_w in set(w for s in m for w in s))
    vocab = vocab.union(set(v_w.strip(string.punctuation).lower() for m in queries for v_w in set(w for s in m for w in s)))
    vocab = vocab.union(set(a[0].strip(string.punctuation).lower() for a in ans))
    return vocab





def encoder(mems, queries, ans, emb_dims, MAX_MEM_LEN, MAX_SEN_LEN, vocab):


    encoded_qs, encoded_mems, encoded_as = [], [], []
    for q_i, m_i, a_i in zip(queries, mems, ans):

        #encode questions
        q_hot_text = one_hot(' '.join(q_i[0]).strip(string.punctuation).strip('\n'), len(vocab))
        q_hot_text += ([0] * (MAX_SEN_LEN - len(q_hot_text)))
        encoded_qs.append(q_hot_text)     

        #encode memories
        m_hot_texts = []
        for m in m_i:
            m_sen_enc = one_hot(' '.join(m).strip(string.punctuation).strip('\n'), len(vocab))
            m_sen_enc += ([0] * (MAX_SEN_LEN - len(m_sen_enc))) 
            m_hot_texts.append(m_sen_enc)  

        if len(m_hot_texts) < MAX_MEM_LEN:
            m_hot_texts = np.append(m_hot_texts, [([0] * MAX_SEN_LEN) for _ in range(MAX_MEM_LEN - len(m_hot_texts))], axis=0)

        encoded_mems.append(m_hot_texts)
        
        #encode answers
        encoded_as.append(one_hot(' '.join(a_i).strip(string.punctuation).strip('\n'), len(vocab)))
        
    
    return encoded_qs, encoded_mems, encoded_as

  
  
  
  
#position encoder
def pe(mems):
    
    pe_vec = np.zeros((MAX_SEN_LEN, emb_dims))
    for pos in range(1, MAX_SEN_LEN + 1):
        for k in range(1, emb_dims + 1):
            v = (1 - pos / MAX_SEN_LEN) - (k / emb_dims) * (1 - (2 * pos / MAX_SEN_LEN))
            pe_vec[pos - 1][k - 1] = v

    
    pe_vec = K.variable(pe_vec)
 
    pe_vec = K.expand_dims(pe_vec, axis=0)
    pe_vec = K.expand_dims(pe_vec, axis=0)
    pe_vec = K.repeat_elements(pe_vec, MAX_MEM_LEN, axis=1)
    pe_vec = K.repeat_elements(pe_vec, batch_size, axis=0)
    
    
    return mems * pe_vec







#time encoders, implements Random Noise of 10% zeros to TA or TC if RN True in model parameters      
class TimeLayer(keras.layers.Layer):
    
    def __init__(self, t, RN, MAX_MEM_LEN, emb_dims, **kwargs):
        self.t = t
        self.MAX_MEM_LEN = MAX_MEM_LEN
        self.emb_dims = emb_dims
        super(TimeLayer, self).__init__(**kwargs)    
        
    def build(self, input_shape):
        if self.t == "TA":
            self.W = self.add_weight(shape=(input_shape[1], input_shape[2]), initializer='truncated_normal', name="TA")
        elif self.t == "TC":
            self.W = self.add_weight(shape=(input_shape[1], input_shape[2]), initializer='truncated_normal', name="TC")
        super().build((self.MAX_MEM_LEN, self.emb_dims))

    def call(self, inputs):
        
        if RN:
            w_size = np.prod(K.int_shape(self.W))     #randomly mask out 10% of TA, TC layers per 4.1
            noise = np.ones(w_size)
            noise[:int(w_size * 0.1)] = 0
            np.random.shuffle(noise)
            noise = noise.reshape(self.W.shape)
            T = inputs + self.W * noise
        else:
            T = inputs + self.W
        
        return T
    
    

    
#weights shared as described in 4.1 (only initially here though??)
def loss_init(args):
    return K.transpose(A[-1].weights[0])


def custom_loss(u):
    W =  K.transpose(A[-1].weights[0])
    out = K.dot(u, W)
    return K.softmax(out)







def scheduler(epoch):
    if epoch%50==0 and epoch!=0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.95)
        print(f"lr changed to {lr*.95}")
    return K.get_value(model.optimizer.lr)





    
    

  
  
  
  
