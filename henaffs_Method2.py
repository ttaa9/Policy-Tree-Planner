from forwardModel3 import *
from SimpleTask import SimpleGridTask
import numpy as np, numpy.random as npr, random as r, SimpleTask
from TransportTask import TransportTask
from NavTask import NavigationTask
import tensorflow as tf
import os
import time

class Henaff_Planning():
    def __init__(self,num_actions,len_action, len_state, iterations, gamma):
        self.initialize_forward_model()
        self.x= tf.Variable(tf.random_normal([num_actions,len_action])*0.1,name='x', dtype=tf.float32)
        self.s_0= tf.placeholder(tf.float32, [len_state])
        self.s_f= tf.placeholder(tf.float32, [len_state])
        self.iterations=iterations
        self.num_actions = num_actions
        self.len_action = len_action
        self.len_state = len_state
        self.gamma=gamma
        self.lr_interval=[0.5,0.5]
        self.eps=tf.constant(0.00001,shape=[self.num_actions,self.len_action])
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "x")
        
    def initialize_forward_model(self):
        self.fm=ForwardModel(64,74,10, 100)
        self.fm.load_model('abcd.ckpt')
        
    def optimize(self ,init_state,final_state, env):
        sess = tf.get_default_session()
        for i in range(0,self.iterations):
            lr=self.lr_interval[1]-i*((self.lr_interval[1]-self.lr_interval[0])/self.iterations)
            print('learning rate: ',lr)

            epsilon=tf.random_normal([self.num_actions,self.len_action])*self.gamma
            self.x=self.x+epsilon
            self.a=tf.nn.softmax(self.x)
    
            current_state=self.s_0
            state_out = self.fm.get_initial_features(1)
            for t in range(0,self.num_actions):
                
                #print(t)
                current_state=tf.reshape(current_state,[64,])
                concat_vec = tf.concat([tf.cast(current_state,dtype=tf.float32),self.a[t]],axis=0)
                concat_vec=tf.reshape(concat_vec,[1,1,-1]) #[batch size, sequence length, size of concat_vec]
                
                current_state,state_out = self.fm.dynamic_cell(concat_vec,tf.constant([1]), state_out)
                c, h = state_out
                state_out= tf.nn.rnn_cell.LSTMStateTuple(c, h)
        
            
            current_state=tf.reshape(current_state,[64,-1])
            predVecs = env.deconcatenateOneHotStateVector(current_state)
            labelVecs = env.deconcatenateOneHotStateVector(self.s_f)
          
            loss=0
            for pv,lv in zip(predVecs,labelVecs):
                loss += tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(pv), labels=lv)

            print("Iteration: ####################################### ", i)
            print(tf.global_variables())

            self.gradients = tf.gradients(loss, [self.x])
            print(self.gradients)
            self.x = self.x - tf.multiply(self.gradients[0], lr) 
            
            print("grad",tf.report_uninitialized_variables())

            loss, value_x, s_curr= sess.run([loss, self.x, current_state],{self.s_0:init_state,self.s_f:final_state})
            print(loss)            
            for xi in value_x:
                print(NavigationTask.actions[np.argmax(xi)])
            print(loss)
            scurrv = env.deconcatenateOneHotStateVector(s_curr)
            print('px:',np.argmax(scurrv[0]))
            print('py:',np.argmax(scurrv[1]))
            print('d:',np.argmax(scurrv[2]))
            print('gx:',np.argmax(scurrv[3]))
            print('gy:',np.argmax(scurrv[4]))
            




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
            hp = Henaff_Planning(10,10,64,20,0.0000001)#initialize hennaff planning method
            init = tf.variables_initializer(hp.trainable_vars)
            sess.run(init)
            print(state_i,state_f)
            action_sequence=hp.optimize(state_i,state_f,env)

        #convert action sequence to [num_action,] action numer ids
        action_sequence=np.argmax(action_sequence,1)
        action_sequence=np.reshape(action_sequence,[len(action_sequence),])

        for action in action_sequence:
            print('\n')
            print('-Initial State-')
            env.display()
            print('-Action Taken-')
            env.performAction(action)
            print(env.actions[action])
            print('-Resultant State-')
            env.display()

if __name__ == '__main__':
    navmain()