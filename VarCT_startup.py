#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hannesmeinlschmidt
"""

import numpy as np
from numpy.random import default_rng
from scipy.integrate import odeint
from scipy.linalg import expm
# from scipy import stats
# import matplotlib
import matplotlib.pyplot as plt
import time
import argparse
import csv
import sdeint

from VarCT_processdata import *

# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("-td", "--trainingdata", help="Percentage of panel data used", default="1", type=float)
parser.add_argument("-li", "--learnintercepts", help="Learn intercepts", default=False, action="store_true")
parser.add_argument("-all", "--learn_all", help="Learn intercepts for every subject in panel data", default=False, action="store_true")
parser.add_argument("-sim", "--simulate", help="Simulate data", default=False, action="store_true")
parser.add_argument("-n", "--num_ivs", help="Simulate many trajectories", default="50", type=int)
parser.add_argument("-t", "--timehorizon", help="Simulate time horizon", default="15", type=int)
parser.add_argument("-ni", "--num_ints", help="Number of intercept classes in simulation", type=int)
parser.add_argument("-np", "--num_plots", help="Number of trajectories plotted", default="50", type=int)
parser.add_argument("-ex", "--exportdata", help="Export simulated data", default=False, action="store_true")
args = parser.parse_args()


# Pretty matrix print
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


# Parameters in objective function (Tychonov/regularization)
beta_A = 0
# beta_A = 1
beta_F = 0
# beta_F = 10**(-5)


# Differential equation functions
def ode_f(vec, t, mat, f):
    mat = mat.reshape(M, M)
    out = np.dot(mat, vec) + f
    return out


def adj_f(vec, t, mat):
    out = -mat.T.dot(vec)
    return out


# SDE noise for simulation, adjust size to taste
def G(x, t):
    return np.diag([0.5, 1])


# General parameters
rng = default_rng()
# Tolerance for odeint solver
ode_tol = 1.49012e-6
# Time discretization steplength
step = 1e-1
time_horizon = args.timehorizon

# Matrix produced by CTSEM for later comparisons
A_CTSEM = np.array([[-0.4570244, 0.2622246], [0.3762264, -0.5774997]])
# Offsets in CTSEM
t0_CTSEM = np.array([0, 0])

learn_intercepts = args.learnintercepts

#
# Globally expected variables to be filled from either panel data or simulation
# ivs -- list of M-tuples with the initial values for each trajectory/subject
# targets -- list of M-tuples with panel data/trajectory evaluations at certain time points
# eval_indices -- list of indices such that times[eval_indices[i][:]] corresponds to targets time points
# times_iv -- list of time discretization lists starting from the respective starting time
#

print("------------ STARTUP ------------")
print(time.ctime(time.time()))

# Parse panel data
if not args.simulate:
    perc = args.trainingdata*(10**(-2))
    print("Using panel data")
    print("Choose " + str(args.trainingdata) + "% of data sets randomly")
    
    # See VarCT_processdata.py
    years, values, time_horizon = processPanelData()
    
    # Pick random subset of size indicated in parameters
    num_ivs = int(np.ceil(len(years)*perc))
    rng = default_rng()
    subset = np.sort(rng.choice(len(years), size=num_ivs, replace=False))
    time_eval = [years[i] for i in subset]
    targets = [values[i] for i in subset]
    M = len(targets[0][0])
    ivs = [target[0] for target in targets]

    # Learn intercepts logic
    learn_num = num_ivs
    learn_class = range(num_ivs)
    f_class = range(num_ivs)
    # Either learn intercepts for each individual ...
    if learn_intercepts:
        # ... or learn one intercept for all individuals
        if not args.learn_all:
            learn_num = 1
            learn_class = np.zeros(num_ivs)
            f_class = np.zeros(num_ivs)

    # Dummy variable in this case
    f_true = np.tile(t0_CTSEM, (learn_num, 1))

# Time discretization list
times = np.arange(0, time_horizon + step, step)
    
# Prepare data simulation
if args.simulate:
    print("Simulate data")
    M = 2
    num_ivs = args.num_ivs
    ivs = []
    # List of possible time evaluation points
    time_evals_all = [0, 1, 2, 3, 4, 5, 10, 20, times[-1]]
    # Remove possible duplicate
    time_evals_all = list(dict.fromkeys(time_evals_all))
    time_evals_list = [eval for eval in time_evals_all if eval <= times[-1]]
    time_eval = []
    box = 5
    for _ in range(num_ivs):
        # Pick random initial values in [-5,5]^M
        iv = 2 * box * np.random.random_sample(M) - box
        ivs.append(iv)
        # Pick time evaluation points at random (but at least 2)
        how_many = np.random.randint(2, len(time_evals_list))
        time_eval.append(np.sort(np.random.choice(time_evals_list, how_many, replace=False)))

    # If no number of intercept classes is given, every individual has its own
    if args.num_ints is None:
        f_num = num_ivs
    else:
        f_num = args.num_ints

    # If number of intercept classes < number of individuals, assign random intercept class
    f_class = range(f_num)
    if num_ivs > f_num:
        f_class = np.append(f_class, rng.choice(f_num, size=num_ivs-f_num, replace=True))

    # Learn intercepts logic
    if args.learn_all:
        learn_num = num_ivs
        learn_class = range(num_ivs)
    else:
        learn_num = f_num
        learn_class = f_class

# Some output to start with
num_plots = np.minimum(args.num_plots, num_ivs)
plot_indices = np.sort(rng.choice(num_ivs, size=num_plots, replace=False))

print("Number of trajectories: " + str(num_ivs))
print("Tychonov parameter for matrix: " + str(beta_A))
print("Time horizon: " + str(time_horizon))
print("Time steps: " + str(step))
if args.simulate:
    print("Number of intercept classes: " + str(f_num))
if learn_intercepts:
    print("Learning intercepts")
    print("Number of intercept classes to learn: " + str(learn_num))
    print("Tychonov parameter for intercepts: " + str(beta_F))
else:
    print("Do not learn intercepts")
print("Plotting " + str(num_plots) + " trajectories")

# Fill globally expected variables
eval_indices = []
# obj_div collects number of targets, used for weighting
obj_div = []
times_iv = []
for j in range(num_ivs):
    times_first = [i for i in range(len(times)) if time_eval[j][0] == times[i]]
    obj_div.append(len(time_eval[j]))
    times_local = times[times_first[0]:]
    times_iv.append(times_local)
    eval_indices_local = [i for i in range(len(times_local)) if times_local[i] in time_eval[j]]
    eval_indices.append(eval_indices_local)


# Simulate data routine
def simulate_data():
    print("Simulating data...")
    start = time.time()
    # noise = 1
    scale = 1

    # Generate random matrix A, negative semidefinite
    A_true = 2 * scale * np.random.random_sample((M, M)) - scale
    while not np.all(np.real(np.linalg.eig(A_true)[0] <= 0)):
        A_true = 2 * scale * np.random.random_sample((M, M)) - scale

    # Easy fixed A
    A_true = np.diag([-1, -2])
    # Generate random intercepts
    f_choice = 2 * box * np.random.random_sample((f_num, M)) - box
    f_true = -f_choice.dot(A_true.T)
    
    # If data to export for comparison in CTSEM, normalize to intercept 0
    if args.exportdata:
        f_true = f_true-f_true
        
    # Generate targets
    targets = []
    trajectories_sim = []
    for k, iv in enumerate(ivs):
        fk = f_true[int(f_class[k])]

        def ode_fA(x, t):
            return ode_f(x, t, A_true, fk)

        # Use deterministic integrator odeint + noise or stochastic integrator sdeint to create trajectories
        # Deterministic
        # trajectory = odeint(ode_fA, iv, times_iv[k], atol=ode_tol, rtol=ode_tol)
        # ... + noise
        # trajectory_noise = trajectory + noise*np.random.random_sample((len(trajectory), M))-(noise/2)
        # Stochastic
        trajectory = sdeint.itoint(ode_fA, G, iv, times_iv[k])
        trajectory_noise = trajectory

        targets.append(trajectory_noise[eval_indices[k]])
        # Unfortunately we need to save all simulated trajectories in the case of SDE simulation
        # since we cannot reproduce it just from A and the initial value
        trajectories_sim.append(trajectory_noise)

    end = time.time()
    print("... done in {:1.2f}s!".format(end-start))
    return targets, A_true, f_true, trajectories_sim


if args.simulate:
    targets, A_true, f_true, trajectories_sim = simulate_data()
    if args.exportdata:
        print("A_true:")
        matprint(A_true)
        print(np.linalg.eig(A_true)[0])
        with open('export_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['pid', 'syear', 'plh0171', 'plh0173'])
            for k, targetlist in enumerate(targets):
                for j, targ in enumerate(targetlist):
                    pid = k+1
                    syear = int(time_eval[k][j]+1984)
                    plh0171 = targ[0]
                    plh0173 = targ[1]
                    writer.writerow([pid, syear, plh0171, plh0173])


print("---------- STARTUP DONE -----------")
print("")
