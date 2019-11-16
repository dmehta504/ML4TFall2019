"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: dmehta32 (replace with your User ID)
GT ID: 902831571 (replace with your GT ID)
"""

import numpy as np
import random as rand


class QLearner(object):

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.action_rate = rar
        self.action_decay_rate = radr
        self.dyna = dyna
        self.verbose = verbose

        self.q_table = np.random.uniform(-1.0, 1.0, size=(self.states, self.num_actions))
        self.experience_dict = {'s': [], 'a': [], 's_prime': [], 'r': []}

    def querysetstate(self, s):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """
        self.s = s

        if rand.random() < self.action_rate:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[self.s])

        if self.verbose: print(f"s = {s}, a = {action}")
        return action

    def query(self, s_prime, r):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """
        # Update Q-Table according to equation
        # 1 - alpha * Q[s, a] + alpha * (r + gamma * max(Q[s_prime]))
        alpha = self.alpha
        gamma = self.gamma
        best_action = np.argmax([self.q_table[s_prime]])
        self.q_table[self.s, self.a] = (1 - alpha) * self.q_table[self.s, self.a] + \
                                       alpha * (r + gamma * (self.q_table[s_prime, best_action]))

        # Update experience
        self.experience_dict['s'].append(self.s)
        self.experience_dict['a'].append(self.a)
        self.experience_dict['s_prime'].append(s_prime)
        self.experience_dict['r'].append(r)

        # Execute Dyna if specified
        if self.dyna > 0:
            memory = len(self.experience_dict['s'])
            random_choices = np.random.randint(memory, size=self.dyna)
            for i in range(self.dyna):
                choice = random_choices[i]
                s = self.experience_dict['s'][choice]
                a = self.experience_dict['a'][choice]
                r = self.experience_dict['r'][choice]
                sprime = self.experience_dict['s_prime'][choice]
                best_a = np.argmax([self.q_table[sprime]])
                self.q_table[s, a] = (1 - alpha) * self.q_table[s, a] + alpha * (r + gamma * self.q_table[sprime, best_a])

        # Decide the best action to take based on the action rate
        if rand.random() < self.action_rate:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = best_action

        # Update action, current state and decay the action rate
        self.a = action
        self.s = s_prime
        self.action_rate = self.action_rate * self.action_decay_rate

        if self.verbose: print(f"s = {s_prime}, a = {action}, r={r}")
        return action

    def author(self):
        return 'dmehta32'


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
