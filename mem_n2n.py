
from keras.layers import Input, Embedding, Lambda, Activation, Dense, Reshape, RepeatVector, Dropout
from keras.callbacks import LearningRateScheduler, TensorBoard, Callback
from keras.preprocessing.text import one_hot
import matplotlib.pyplot as plt
from task_prep import task_eval
from keras import backend as K 
from keras.models import Model
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import random
import string
import keras


# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)  #to avoid printing all the warnings about deprecations, etc.


#parser = argparse.ArgumentParser()
#parser.add_argument('--task_num', type=int, default=1, help='default is task 1')
#args = parser.parse_args()

#global g_task_num
#g_task_num = args.task_num
g_task_num = 1
assert g_task_num >= 1 and g_task_num <= 20


#parser.add_argument('--embd_size', type=int, default=30, help='default 30. word embedding size')
#parser.add_argument('--batch_size', type=int, default=32, help='default 32. input batch size')
#parser.add_argument('--n_epochs', type=int, default=100, help='default 100. the number of epochs')
#parser.add_argument('--max_story_len', type=int, default=25, help='default 25. max story length. see 4.2')
#parser.add_argument('--use_10k', type=int, default=1, help='default 1. use 10k or 1k dataset')
#parser.add_argument('--test', type=int, default=0, help='defalut 1. for test, or for training')
#parser.add_argument('--resume', type=int, default=1, help='defalut 1. read pretrained models')
#parser.add_argument('--seed', type=int, default=1111, help='random seed')
#args = parser.parse_args()







#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
#tf.logging.set_verbosity(tf.logging.ERROR)  #to avoid printing all the warnings about deprecations, etc.

global GLOBAL_LOSS_DECREASING 
global STARTED


GLOBAL_LOSS_DECREASING = True
STARTED = False

    
#see 4.2 of Sukhbaatar, et al. (2015), helps avoid getting stuck in local minima from bad initializations
class LinearStart(keras.callbacks.Callback):
    
    def __init__(self):
        self.FIRST = True
        self.MOD_STARTED = False
        self.losses = [999]
        super().__init__()
 
    def on_batch_end(self, batch, logs={}):
        val_loss = logs.get('val_loss')
        if not self.MOD_STARTED:
            if not self.FIRST and val_loss >= self.losses[-1]:
                GLOBAL_LOSS_DECREASING = False
                STARTED = True
            FIRST = False
            self.MOD_STARTED = True
        self.losses.append(val_loss)
        return


    
#time encoders, implements Random Noise of 10% zeros to TA or TC if RN True in model parameters      
class TimeLayer(keras.layers.Layer):
    
    def __init__(self, t, RN, MAX_MEM_LEN, emb_dims, **kwargs):
        self.t = t
        self.RN = RN
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
        
        if self.RN:
            w_size = np.prod(K.int_shape(self.W))     #randomly mask out 10% of TA, TC layers per 4.1
            noise = np.ones(w_size)
            noise[:int(w_size * 0.1)] = 0
            np.random.shuffle(noise)
            noise = noise.reshape(self.W.shape)
            T = inputs + self.W * noise
        else:
            T = inputs + self.W
        
        return T

    
    
    

class MemN2N():
    def __init__(self, 
                 task_num,
                 emb_dims, 
                 n_hops, 
                 n_epochs, 
                 batch_size, 
                 dropout, 
                 val_split, 
                 verbose=2, 
                 time_embed=True, 
                 pos_encode=True,
                 LS = False,
                 RN = False
                ):
    
    
        #parameters for the model 
        self.emb_dims = emb_dims  #dimension of word embeddings
        self.n_hops = n_hops
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.time_embed = time_embed
        self.pos_encode = pos_encode
        self.dropout = dropout
        self.val_split = val_split
        self.verbose = verbose
        self.LS = LS
        self.RN = RN
        
        
        #DATA
        (self.tr_mems, self.tr_qs, self.tr_ans), (self.ts_mems, self.ts_qs, self.ts_ans) = task_eval(task_num)
        
        
        
        #print random sample
        print("NOT RANDOM SAMPLE:")
        rint = random.randint(1, len(self.tr_mems))
        print(self.tr_mems[0], self.tr_qs[0], self.tr_ans[0])
        print(self.ts_mems[0], self.ts_qs[0], self.ts_ans[0])
        
     
        
        self.vocab = self.get_vocab()
        
        
        if 0:  #self.LS:
            self.MAX_MEM_LEN = 50       #4.2
        else:
            self.MAX_MEM_LEN = max([len(m) for m in self.tr_mems + self.ts_mems]) 
        
        self.MAX_SEN_LEN = max([len(s) for m in self.tr_mems + self.tr_qs for s in m])
       
    
        #one hot encoded forms of the data, 
        #i.e. each sentence is a row of word embeddings, weighted by the position of the word
        #TODO: implement using pretrained embeddings
        self.tr_encoded_mems, self.tr_encoded_qs, self.tr_encoded_ans = self.old_encoder(self.tr_mems, self.tr_qs, self.tr_ans) 
        self.ts_encoded_mems, self.ts_encoded_qs, self.ts_encoded_ans = self.old_encoder(self.ts_mems, self.ts_qs, self.ts_ans)
                                                                                   
       
        self.lr_decay = LearningRateScheduler(self.scheduler)
        #self.tb_graph = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        self.ls_call  = LinearStart()
        
        self.model = self.build()
        
        print(self.model.summary())
        
        

    def train(self):
        return self.model.fit([self.tr_encoded_mems, self.tr_encoded_qs], np.concatenate(self.tr_encoded_ans),
                              shuffle=True,
                              epochs=self.n_epochs,  
                              batch_size=self.batch_size,
                              verbose=self.verbose,
                              validation_split=self.val_split,
                              callbacks=[self.lr_decay] #add back tb for cool graphs/stats
                             )

    def test(self):
        return self.model.evaluate([self.ts_encoded_mems, self.ts_encoded_qs], 
                            np.concatenate(self.ts_encoded_ans), 
                            verbose=self.verbose, 
                            batch_size=self.batch_size
                           )

    
    def build(self):

        #declaring the form of the input layers
        x_mems = Input(batch_shape=(self.batch_size, self.MAX_MEM_LEN, self.MAX_SEN_LEN), name='mem_input')
        x_qs = Input(batch_shape=(self.batch_size, self.MAX_SEN_LEN), name='q_input')


        #A, B memory and question embeddings, respectively
        A = [Embedding(
            len(self.vocab) + 1, self.emb_dims, embeddings_initializer='truncated_normal', mask_zero=True
            ) for _ in range(self.n_hops + 1)]
        
        B = A[0]

        
        TA = TimeLayer("TA", self.RN, self.MAX_MEM_LEN, self.emb_dims)
        TC = TimeLayer("TC", self.RN, self.MAX_MEM_LEN, self.emb_dims)


        u = Dropout(0.1)(Lambda(lambda args: K.sum(B(args), axis=1), name='embedding_sum')(x_qs))

        #utility functions
        add = Lambda(lambda args: args[0] + args[1], name='add')
        sum_2 = Lambda(lambda args: K.sum(args, axis=2), name='sum_2')



        for k in range(self.n_hops):

            m = A[k](x_mems)   #embed the encoded memories
            c = A[k+1](x_mems)

            if self.time_embed:

                m = Lambda(self.pe, name="pos_encoding_hop" + str(k))(m)   #positionally encode the embeddings
                m = sum_2(m)                                          #collapse into single d x |V| vector 
                c = sum_2(c)                                          #collapse into single d x |V| vector 

                m = TA(m)                                              #apply time information
                c = TC(c)

            else:
                m = sum_2(m)
                c = sum_2(c)


            u_map = RepeatVector(m.shape[1])(u)                      #scale q vector for memory size

            p = Lambda(lambda args: args[0] * args[1])([m, u_map])   #apply similarity function between mem, q for each pair
            p = sum_2(p)
            
            if self.LS:
                if GLOBAL_LOSS_DECREASING or STARTED:
                    p = Activation('softmax')(p)                             #compute distribution of similarity 
            else:
                p = Activation('softmax')(p)  
                    
            p = Reshape((1, p.shape[1]))(p)

            o = Lambda(lambda args: K.batch_dot(args[0], args[1]), name="batch_mul_hop" +str(k))([p, c])  #apply sim to mems
            o = Lambda(lambda x: K.sum(x, axis=1), name="collapse_o_hop" + str(k))(o)                    #collapse to d x |V|

            u = add([u, o])                                          #apply to embedded q vector at end of each hop

            
        
        
        self.W = A[-1].weights[0]                                 #for use in loss_init
        
        
        #output for network (p(each word in vocab|(mems, q)))
        out = Lambda(self.custom_loss, name="out_layer")(u)
        

        model = Model(inputs=[x_mems, x_qs], outputs=out)
    
    
        #sparse_cat_cross.. allows the y vector to not have to be made into one-hot vectors 
        #(integers representing single target word)
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model



    #weights shared as described in 4.1 (only initially here though??)
    def loss_init(self, dummy_args):
        return K.transpose(self.W)

    def custom_loss(self, u):
        W =  K.transpose(self.W)
        out = K.dot(u, W)
        return K.softmax(out)
        


    def scheduler(self, epoch):
        if epoch%100==0 and epoch!=0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr*.95)
            print(f"LR changed to {lr*.95}")
        elif epoch == 0:
            if self.LS:
                print(f"LR initialized to: {0.005}, for LS")
                K.set_value(self.model.optimizer.lr, 0.005)
            else:
                print(f"LR initialized to: {0.01}")
                K.set_value(self.model.optimizer.lr, 0.01)
        return K.get_value(self.model.optimizer.lr)

    
    
    
    

    def get_vocab(self):
        vocab = set(v_w.strip(string.punctuation).lower() for m in self.tr_mems for v_w in set(w for s in m for w in s))
        vocab = vocab.union(
            set(v_w.strip(string.punctuation).lower() for m in self.tr_qs for v_w in set(w for s in m for w in s))
            )
        vocab = vocab.union(set(a[0].strip(string.punctuation).lower() for a in self.tr_ans))
        return vocab

    
    
    def old_encoder(self, mems, queries, ans):


        encoded_qs, encoded_mems, encoded_as = [], [], []
        for q_i, m_i, a_i in zip(queries, mems, ans):

            #encode questions
            q_hot_text = one_hot(' '.join(q_i[0]).strip(string.punctuation).strip('\n'), len(self.vocab))
            q_hot_text += ([0] * (self.MAX_SEN_LEN - len(q_hot_text)))
            encoded_qs.append(q_hot_text)     

            #encode memories
            m_hot_texts = []
            for m in m_i:
                m_sen_enc = one_hot(' '.join(m).strip(string.punctuation).strip('\n'), len(self.vocab))
                m_sen_enc += ([0] * (self.MAX_SEN_LEN - len(m_sen_enc))) 
                m_hot_texts.append(m_sen_enc)  

            if len(m_hot_texts) < self.MAX_MEM_LEN:
                m_hot_texts = np.append(m_hot_texts, [([0] * self.MAX_SEN_LEN) for _ in range(self.MAX_MEM_LEN - len(m_hot_texts))], axis=0)

            encoded_mems.append(m_hot_texts)

            #encode answers
            encoded_as.append(one_hot(' '.join(a_i).strip(string.punctuation).strip('\n'), len(self.vocab)))


        return encoded_mems, encoded_qs, encoded_as

    
    
    
    
    

    def encoder(self, mems, queries, ans):
        
        for i, (q_i, m_i, a_i) in enumerate(zip(queries, mems, ans)):
         
            #encode questions
            q_hot_text = np.array(one_hot(' '.join(q_i[0]).strip(string.punctuation).strip('\n'), len(self.vocab)))
            q_hot_text = np.append(q_hot_text, [0] * (self.MAX_SEN_LEN - len(q_hot_text)))
          
            if i == 0:
                encoded_qs = q_hot_text
            else:
                encoded_qs = np.vstack((encoded_qs, q_hot_text))     

          
            #encode memories
            for j, m in enumerate(m_i[:min(len(m_i), self.MAX_MEM_LEN)]):
                
                m_sen_enc = one_hot(' '.join(m).strip(string.punctuation).strip('\n'), len(self.vocab))
                m_sen_enc = np.append(m_sen_enc, [0] * (self.MAX_SEN_LEN - len(m_sen_enc)))  #pad sentences
                
                if j == 0:
                    m_hot_texts = m_sen_enc
                else:
                    m_hot_texts = np.vstack((m_hot_texts, m_sen_enc))
            
            
             
            if len(m_hot_texts.shape) == 1:
                m_hot_texts = np.expand_dims(m_hot_texts, axis=0)
            
            #pad mems
            if m_hot_texts.shape[0] < self.MAX_MEM_LEN:
                pad = np.zeros((self.MAX_MEM_LEN - m_hot_texts.shape[0], self.MAX_SEN_LEN))
                m_hot_texts = np.append(m_hot_texts, pad, axis=0)
                   
            #having the filter off this embedding means we can treat the sequences as single words (better to have seq2seq...)
            #also this is disingenuous and should never be used on smaller datasets without expanding the vocab space to be
            #at least as large as all the noun combinations, otherwise we only match combos the model has seen....
            m_ans = np.array(one_hot(' '.join(a_i).strip(string.punctuation).strip('\n'), len(self.vocab), filters=[]))
            
    
            if i == 0:
                encoded_mems = np.expand_dims(m_hot_texts, axis=0)
                encoded_ans = m_ans
            else:
                encoded_mems = np.append(encoded_mems, [m_hot_texts], axis=0)
                encoded_ans = np.vstack((encoded_ans, m_ans))
            
            

        print("ENCODE MEMs, Qs, As [0]")
        print(encoded_mems[0], encoded_qs[0], encoded_ans[0])
        return encoded_mems, encoded_qs, encoded_ans



    #position encoder
    def pe(self, mems):

        pe_vec = np.zeros((self.MAX_SEN_LEN, self.emb_dims))
        for pos in range(1, self.MAX_SEN_LEN + 1):
            for k in range(1, self.emb_dims + 1):
                v = (1 - pos / self.MAX_SEN_LEN) - (k / self.emb_dims) * (1 - (2 * pos / self.MAX_SEN_LEN))
                pe_vec[pos - 1][k - 1] = v


        pe_vec = K.variable(pe_vec)  #to make sure its weight's get trained

        pe_vec = K.expand_dims(pe_vec, axis=0)
        pe_vec = K.expand_dims(pe_vec, axis=0)   #to make a (1, 1, MAX_SEN_LEN, emb_dims),
        pe_vec = K.repeat_elements(pe_vec, self.MAX_MEM_LEN, axis=1) #for scaling first to this mem dimension,
        pe_vec = K.repeat_elements(pe_vec, self.batch_size, axis=0)  #then to the batch dim


        return mems * pe_vec   #apply the encoding transformation
    

def main():

   
    histories = []
    accuracies = []

    end = g_task_num #modify to span tasks
    
    for i in range(g_task_num, end+1):
        print(f'\n*********Task num: {i}*********')
        m = MemN2N(task_num=i,   #1 - 20
               emb_dims=20, 
               n_hops=3, 
               n_epochs=100, 
               batch_size=50, 
               dropout=0.1, 
               val_split=0.1,
               LS=False, 
               RN=False,
               verbose=1

            )

        history = m.train()
        test_accuracy = m.test()[1]
        print(f"Accuracy on test set: {test_accuracy}")

        pd.DataFrame(history.history).plot(title=f"Task No. {i}");
 
        plt.show()
        histories.append(history.history)
        accuracies.append(test_accuracy)

    data = {'Task':[str(i) for i in range(1, len(accuracies) + 1)], 'Accuracy':accuracies}
    data = pd.DataFrame(data)
    print(data)
    
    #data.to_csv('memn2n_testing_data.csv')
    
    #history_df = pd.DataFrame(histories)
    #history_df.to_csv('training_histories.csv')
    
    
    return data


data = main()
       

   
    
    
    
    
    
    



