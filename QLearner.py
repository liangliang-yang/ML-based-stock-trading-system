import numpy as np
import random as rand

class QLearner(object):
    
    def author(self):
        return 'lyang338'

    def __init__(self, \
        num_states=1000, \
        num_actions = 3, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        outrar = 0.5,\
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q = np.zeros([num_states, num_actions])
        self.Tc = 0.000001*np.ones((num_states, num_actions, num_states))
        self.T =  (1/num_states)*np.ones((num_states, num_actions, num_states))
        self.R = np.zeros((num_states, num_actions))


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        #print 's', s
        self.s = s
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.q[self.s])
        self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action

    def reset_rar(self):
        self.rar = 0.1
        return self.rar
        
    
    
    def update(self, state, action, reward, nextstate, alpha, gamma):
        #print 'max', max(self.q[nextstate]), gamma, alpha, alpha*( reward + gamma*max(self.q[nextstate]) -  self.q[state][action])
        #print self.q
        #print 'before', self.q[state][action]
        #self.q[state][action] += 0
        self.q[state][action] += alpha*( reward + gamma*max(self.q[nextstate]) -  self.q[state][action])
        #print state, action, reward, nextstate
        #print 'after', self.q[state][action]
        #print self.q
        return self.q[state][action]

    def queryQ(self, s, a):
        return self.q[s,a]

    def returnQ(self):
        return self.q

    
    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        # Update Q-table
        #print 'before update'
        #print self.q
        
        self.q[self.s, self.a] = self.update(self.s, self.a, r, s_prime, self.alpha, self.gamma)

        #print 'after update'
        #print self.q

                
        # choose action
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.q[s_prime])
        self.a = action
        self.s = s_prime
        self.rar = self.rar * self.radr
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
