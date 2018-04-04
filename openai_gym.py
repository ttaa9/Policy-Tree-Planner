import gym
import pdb
env = gym.make('MsPacman-v0')
env.reset()
print(env.action_space)




class opengym_api():

	def __init__(self,
				env_name='Ant-v2',				
				):

		self.replay_buffer=[]
		self.env=gym.make(env_name)


	def random_tradj(self,num_steps,render_bool=False,verbose=False):

		i=0
		env_cnt=0

		self.env.reset()

		while i<num_steps:

			if render_bool:
				self.env.render()

			#get random sample
			action=self.env.action_space.sample()

			if env_cnt==0: #if first step in the env
				observation, reward, done, info = self.env.step(action) #ger action, reward, and observation
				env_cnt+=1
			if env_cnt>=1:
				#[[state_i,a],state_f]
				observation_i=observation
				observation, reward, done, info = self.env.step(action) #ger action, reward, and observation
				env_cnt+=1
				if info['ale.lives']<3:
					self.env.reset()
					env_cnt=0
				else:
					pdb.set_trace()
					self.replay_buffer.append([[observation_i,action],observation])
					i+=1



env=opengym_api()

env.random_tradj(10,verbose=True)
pdb.set_trace()







