from forwardModel3 import *
from SimpleTask import SimpleGridTask
import numpy as np, numpy.random as npr, random as r, SimpleTask
from TransportTask import TransportTask
from NavTask import NavigationTask
import tensorflow as tf
import os
import time
import pdb


class Adam_Optimizer():

    def __init__(self,
                learning_rate_0=0.001,
                beta1=0.9,
                beta2=0.999,
                eps=1e-08,
                use_locking=False):
    
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.lr_0=learning_rate_0
        self.lr=self.lr_0

        self.initialize_ADAMS()

    def step_ADAMS(self,g,x):
        self.t=self.t+1
        self.lr= self.lr*np.sqrt(1-np.power(self.beta2,self.t))/(1-np.power(self.beta1,self.t))
        self.m= self.beta1*self.m + (1-self.beta1)*g
        self.v= self.beta2*self.v + (1-self.beta2)*g*g

        x= x - self.lr*self.m/(np.sqrt(self.v)+self.eps)

        return x

    def initialize_ADAMS(self):
        self.m=0
        self.v=0
        self.t=0
        self.lr=self.lr_0

'''
m_0 <- 0 (Initialize initial 1st moment vector)
v_0 <- 0 (Initialize initial 2nd moment vector)
t <- 0 (Initialize timestep)

t <- t + 1
lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

m_t <- beta1 * m_{t-1} + (1 - beta1) * g
v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)

'''

    

class Henaff_Planning(Adam_Optimizer,):
    def __init__(self,num_actions,len_action, len_state, iterations, gamma):
        

        self.initialize_forward_model()
        self.x= tf.Variable(tf.random_normal([num_actions,len_action]),name='x', dtype=tf.float32)
        self.s_0= tf.placeholder(tf.float32, [len_state])
        self.s_f= tf.placeholder(tf.float32, [len_state])
        self.iterations=iterations
        self.num_actions = num_actions
        self.len_action = len_action
        self.len_state = len_state
        self.gamma=gamma
        


        super().__init__(learning_rate_0=0.001,beta1=0.9,beta2=0.999,eps=1e-08,use_locking=False)

    def initialize_forward_model(self):
        sess = tf.get_default_session()
        
        self.fm=ForwardModel(64,74,10, 100)
        self.fm.load_model('abcd.ckpt')

    def optimize(self,init_state,final_state):
        sess = tf.get_default_session()
        
        #TODO: Need to freeze graph -- https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc ????

        #fm = tf.stop_gradient(fm) not sure how to use stop gradients here 
            #initialize the x real-valued variable function
            
        #self.initialize_ADAMS() #initialized adams optimizer

        #self.x=tf.random_normal([self.num_actions,self.len_action]) # initialize self.x as normal distribrution of mean 0 and std 1
      

        for i in range(0,self.iterations):

            epsilon=tf.random_normal([self.num_actions,self.len_action])*self.gamma
            self.x=self.x+epsilon
            self.a=tf.nn.softmax(self.x)
    
            current_state=self.s_0
            state_out = self.fm.get_initial_features(1)
            #tf.contrib.rnn.LSTMStateTuple(self.c_in,self.h_in)
            for t in range(0,self.num_actions):
                #pdb.set_trace()
                print(t)
                current_state=tf.reshape(current_state,[64,])
                concat_vec = tf.concat([tf.cast(current_state,dtype=tf.float32),self.a[t]],axis=0)
                #concat_vec=tf.tile(concat_vec,[10])
                concat_vec=tf.reshape(concat_vec,[1,1,-1]) #[batch size, sequence length, size of concat_vec]
                
                #current_state,state_out = self.fm.predict(sess.run(concat_vec,feed_dict={self.s_0:init_state}),c,h)
                current_state,state_out = self.fm.dynamic_cell(concat_vec,tf.constant([1]), state_out)
                c, h = state_out
                state_out= tf.nn.rnn_cell.LSTMStateTuple(c, h)
        
            #TODO: is loss distance loss?? Do a custum loss function
            loss = tf.reduce_mean(tf.square(current_state-self.s_f))
            #loss, _ = sess.run([loss,self.x],{self.s_0:init_state,self.s_f:final_state})
            
            #initialize the variables for the first iteration

            #sess.run(self.optimizer, feed_dict={self.input: , self.truevalue:})


            print("Iteration: ####################################### ", i)
            print(tf.global_variables())
            #self.optimize= tf.train.AdamOptimizer(0.01).minimize(loss)
            self.gradients = tf.gradients(loss, [self.x])
            print(self.gradients)
            self.x = self.x - tf.multiply(self.gradients[0], 0.01) 
            # self.gradients= tf.train.AdamOptimizer(0.01).compute_gradients(loss,['x:0'])
            # for i, (grad, var) in enumerate(self.gradients):
            #     if grad is not None:
            #         self.gradients[i] = (grad, var)
            #         print(var)
            # self.optimize= tf.train.AdamOptimizer(0.01).apply_gradients(self.gradients)

            if i == 0:
                init_new_vars_op = tf.initialize_all_variables()
                sess.run(init_new_vars_op)
            #self.x=self.step_ADAMS(grad_x,self.x) #step into gradient descent # looking into partial run s
            loss, value_x= sess.run([loss, self.x],{self.s_0:init_state,self.s_f:final_state})
            print(loss)            

        # input [num_actions, len_actions]--> [num_actions]---> [num_actions, len_actions]
        #a=np.one_hot(np.argmax(np.softmax(value_x),axis=1))



        return value_x




def navmain():

    
    env = NavigationTask() #(stochasticity=0.2)
    

    state_i=env.getStateRep()

    #get goal state
    state_f=env.getStateRep()
    inds = np.cumsum([0,env.w,env.h,len(env.oriens),env.w,env.h])
    state_f[inds[0]:inds[1]] = env._intToOneHot(env.goal_pos[0],env.w)
    state_f[inds[1]:inds[2]] = env._intToOneHot(env.goal_pos[1],env.h)

    state_i=env.getStateRep() #get initial state
    
    #we want the goal state so replce the first position vector with the goal position vector
    with tf.Graph().as_default(), tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        hp = Henaff_Planning(10,10,64,30,0.1)#initialize hennaff planning method
        print(state_i,state_f)
        action_sequence=hp.optimize(state_i,state_f)

    #convert action sequence to [num_action,] action numer ids
    action_sequence=np.argmax(action_sequence,1)
    action_sequence=np.reshape(action_sequence,[len(action_sequence),])

    env.performAction(action_sequence[0])

    env.display()

    print('--')
    print(env.actions[action_sequence[0]])
    env.display()

    pdb.set_trace()
    


if __name__ == '__main__':
    navmain()
