import rebound
import matplotlib.pyplot as plt
import observations
import state
import mcmc
import numpy as np

def run_hmc(Niter, true_state, obs, delt, L, mass):
    hmc = mcmc.Hmc(true_state, obs, delt, L, mass)
    chain = np.zeros((0,hmc.state.Nvars))
    chainlogp = np.zeros(0)
    tries = 0
    for i in range(Niter):
        if(hmc.step()):
            tries += 1
        chainlogp = np.append(chainlogp,hmc.state.get_logp(obs))
        chain = np.append(chain,[hmc.state.get_params()],axis=0)
    print("Acceptance rate: %.2f%%"%((tries/float(Niter))*100))
    return hmc, chain, chainlogp

def create_obs(state, npoint, err, errVar, t):
    obs = observations.FakeObservation(state, Npoints=npoint, error=err, errorVar=errVar, tmax=(t))
    return obs
