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
    def __init__(self, initial_state, obs, delt, l):
        super(Hmc,self).__init__(initial_state, obs)
        self.delta = delt
        self.L = l
        self.momentum_vec = None
        self.old_K = 0.0
        self.new_K = 0.0

    def leap_frog(self):
        prop = self.state.deepcopy()
        self.new_momentum_vec = self.momentum_vec - self.delta*self.state.logp_d*0.5
        for i in range(self.L):
            p = prop.get_params() + self.delta*self.new_momentum_vec
            prop.set_params(p)
            logp, logp_d = prop.get_logp_d(self.obs)
            if(i != self.L-1):
                self.new_momentum_vec = self.new_momentum_vec - self.delta*logp_d
                prop.logp_d = None
        self.new_momentum_vec = self.new_momentum_vec - self.delta*prop.logp_d*0.5
        self.new_momentum_vec = -self.new_momentum_vec
        return prop

    def generate_proposal(self):
        self.state.get_logp_d(self.obs)
        self.momentum_vec = np.random.normal(size=(self.state.Nvars))
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
                self.state.collisionGhostParams = new_state.get_params()
                return False
        self.new_K = np.dot(self.new_momentum_vec, self.new_momentum_vec)
        self.old_K = np.dot(self.momentum_vec, self.momentum_vec)
        if (np.exp(-new_state.logp + self.state.logp - self.new_K + self.old_K) > np.random.uniform()):
            self.state = new_state
            return True
        return False



