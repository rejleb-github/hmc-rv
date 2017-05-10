import numpy as np
import emcee
from scipy import stats
import rebound
from datetime import datetime
import sys
import traceback
import copy

'''
Parent MCMC class
'''
class Mcmc(object):
    def __init__(self, initial_state, obs):
        self.state = initial_state.deepcopy()
        self.obs = obs

    def step(self):
        return True 
    
    def step_force(self):
        tries = 1
        while self.step()==False:
            tries += 1
            pass
        return tries

class Hmc(Mcmc):
    def __init__(self, initial_state, obs, delt, l, masses):
        super(Hmc,self).__init__(initial_state, obs)
        self.delta = delt
        self.L = l
        self.momentum_vec = None
        self.new_momentum_vec = None
        self.old_K = 0.0
        self.new_K = 0.0
        self.mass_vector = masses

    def leap_frog(self):
        prop = self.state.deepcopy()
        assert(len(self.mass_vector) == len(self.momentum_vec))
        minv = np.reciprocal(self.mass_vector)
        self.new_momentum_vec = self.momentum_vec - 0.5*self.delta*np.multiply(minv,-self.state.logp_d)
        for i in range(self.L):
            q = prop.get_params() + np.multiply(minv,self.new_momentum_vec)*self.delta
            #print q
            #print "position(q) updated to ^"
            prop.set_params(q)
            logp, logp_d = prop.get_logp_d(self.obs)
            if(i != self.L-1):
                self.new_momentum_vec = self.new_momentum_vec - self.delta*np.multiply(minv, -logp_d)
                prop.logp_d = None
                #prop.logp = None
        self.new_momentum_vec = self.new_momentum_vec - 0.5*self.delta*np.multiply(minv, -prop.logp_d)
        self.new_momentum_vec = -self.new_momentum_vec
        return prop

    def generate_proposal(self):
        self.state.get_logp_d(self.obs)
        self.momentum_vec = np.random.normal(size=(self.state.Nvars))
        self.new_momentum_vec = 0.0
        proposal_state = self.leap_frog()
        return proposal_state

    def step(self):
        while True:
            try:
                new_state = self.generate_proposal()
                if (new_state.priorHard()):
                    return False
                break
            except rebound.Encounter as err:
                print "Collision! {t}".format(t=datetime.utcnow())
                #self.state.collisionGhostParams = new_state.get_params()
                return False
        massive_matrix = np.linalg.inv(np.diag(self.mass_vector))
        self.new_K = np.dot(self.new_momentum_vec, np.dot(massive_matrix, self.new_momentum_vec))
        self.old_K = np.dot(self.momentum_vec, np.dot(massive_matrix, self.momentum_vec))
        if (np.exp(new_state.logp - self.state.logp - self.new_K*0.5 + self.old_K*0.5) > np.random.uniform()):
            self.state = new_state
            return True
        return False
