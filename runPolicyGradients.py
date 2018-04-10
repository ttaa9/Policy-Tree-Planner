import numpy as np, numpy.random as npr, random as r
import tensorflow as tf  
from NavTask import NavigationTask
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.slim as slim

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

# compute discounted future rewards
def discountedReward(reward, discount_factor = 0.1):
    N = len(reward)
    discounted_rewards = np.zeros(N)
    r =0
    for t in reversed(range(5)):
        # future discounted reward from now on
        r = reward[t] + discount_factor * r
        discounted_rewards[t] = r
    return discounted_rewards

def policyRollout(agent, hparams):
    
    #"Runs one episode"
    episode_length = hparams['epiode_length']
    env = NavigationTask()
    obs, acts, rews = [], [], []
    
    for i in range(0, episode_length): 
        
        state = env.getStateRep(True)
        obs.append(state)
        actionProb, sampleAction  = agent.act_inference(state)
      
        action = actionProb.argmax()
        sampleActionIndex = sampleAction.argmax()
        
        env.performAction(action)
        newState  = env.getStateRep()
        reward = env.getReward() 
    
        values = [action]
        acts.append(np.squeeze(np.eye( hparams['num_actions'])[values]))
        rews.append(reward)
    
    return obs, acts, rews

class SimplePolicy(object):
    
    def __init__(self,obs_space,act_space, h_size=100):
       
        print("Observation Space: " , obs_space)
        print("Action Space: ", act_space)
        
        # Input space: [Episode_length, observations], output:[Episode_Length,action_space]
        self.input = tf.placeholder(tf.float32, [None] + list(obs_space))
        hidden = slim.fully_connected(self.input,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,act_space,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.log_prob = log_prob = tf.log(self.output)
        
        # sample: [Episode_length, action_space]
        self.sample = categorical_sample(self.output, act_space)[0, :]
        
        self.targetAction = tf.placeholder(tf.float32, [None, act_space], name="action")
        self.reward = tf.placeholder(tf.float32, [None], name="reward")
        self.cumaltiveReward = tf.reduce_sum(self.reward)
        
        self.entropy =  tf.reduce_mean(tf.reduce_sum(self.output  * log_prob, 1))
        self.crossEntropy = tf.reduce_sum(log_prob * self.targetAction, 1)
        self.loss = -tf.reduce_mean(self.crossEntropy * self.cumaltiveReward) + 0.4*self.entropy
        self._train = tf.train.AdamOptimizer(0.0003).minimize(self.loss)
        

    def act_inference(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.output, self.sample], {self.input: [ob]})

    def train_step(self, obs, acts, reward):
        sess = tf.get_default_session()
        batch_feed = { self.input: obs, self.targetAction: acts, self.reward: reward}
        return sess.run([self._train, self.loss ], feed_dict=batch_feed)

def main():
    # hyper parameters
    env = NavigationTask()
    input_size = np.shape(env.getStateRep(True))
    hparams = {
            'input_size': input_size,
            'num_actions': 10,
            'epiode_length': 7
    }

    # environment params
    eparams = {
            'num_batches': 100,
            'ep_per_batch': 128
    }

    numIts = 5

    print("Starting Policy Gradient")
    for i in range(numIts):
        print("######################")
        print("Try Number: ", i)
        with tf.Graph().as_default(), tf.Session() as sess:

            pi = SimplePolicy(hparams['input_size'], hparams['num_actions'])

            sess.run(tf.initialize_all_variables())
            
            for batch in range(0, eparams['num_batches']):
                num = 0
                total = 0 
                for i in range(0, eparams['ep_per_batch']):
                    obs, acts, rews = policyRollout(pi, hparams)
                    num += 1 if 1 in rews else 0
                    total += 1
                    pi.train_step(obs, acts, rews)
                if batch%50 == 0:
                    _ , loss = pi.train_step(obs, acts, rews)
                    print("Accuracy ", batch, " : ",str(num/total))


############################
if __name__ == '__main__':
    main()
############################