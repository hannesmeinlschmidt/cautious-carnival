#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hannesmeinlschmidt
"""

from VarCT_startup import *


# Produce list of trajectories from matrix A and intercept list f
def forward_trajectories(A, f):
    trajectories = []
   
    for j, iv in enumerate(ivs):
        trajectory = odeint(ode_f, iv, times_iv[j], tuple([A, f[j]]))
        trajectories.append(trajectory)
    return trajectories


# Produce list of trajectories from matrix A and intercept list f with stochastic noise
def forward_trajectoriesNoise(A, f):
    trajectories = []
   
    for j, iv in enumerate(ivs):
        def local_ode_f(x, t):
            #mat = A.reshape(M, M)
            #return mat.dot(x) + f[j]
            return ode_f(x, t, A, f[j])

        trajectory_noise = sdeint.itoint(local_ode_f, G, iv, times_iv[j])
        trajectories.append(trajectory_noise)
    return trajectories


# Produce list of backwards trajectories from final values and adjoint differential equation
# Used in gradient calculation in VarCT_optimize
def backward_trajectories(fvs, adj_fA):
    trajectories = []
    for j, fvs_j in enumerate(fvs):
        trajectory_j = []
        for i, fv in enumerate(fvs_j):
            times_ft = times_iv[j][0:eval_indices[j][i]+1]
            times_bw = times_ft[::-1]
            trajectory = odeint(adj_fA, fv, times_bw, rtol=ode_tol, atol=ode_tol, full_output=False)
            trajectory_j.append(trajectory[::-1])
        trajectories.append(trajectory_j)
    return trajectories


# Make a long vector into a MxM matrix and a bunch of M-vectors
def devectorize(vec):
    A = vec[:M**2]
    A = np.array(A).reshape(M,M)
    f = []
    for j in range(learn_num):
        f.append(vec[j*M+(M**2):j*M+2+(M**2)])
    return A, f


# Calculate objective function value
def obj_value(errors, A, f, bF=beta_F):
    obj = 0
    for j, error_j in enumerate(errors):
        obj += 0.5 * (obj_div[j]**(-1)) * (np.linalg.norm(error_j)**2)
    obj += 0.5 * beta_A * (np.linalg.norm(A)**2)
    obj += 0.5 * bF * (np.linalg.norm(f)**2)
    return obj


# Some unfortunate detours due to intercept learning regularization
def obj(A, f):
    if len(f) == learn_num:
        rhs = build_rhsFromLearn(f)
    else:
        rhs = build_rhs(f)
    trajectories = forward_trajectories(A, rhs)
    errors = assemble_errors(trajectories)
    obj = obj_value(errors, A, f)
    return obj


def obj_onlyA(A):
    rhs = build_rhs(f_true)
    trajectories = forward_trajectories(A, rhs)
    errors = assemble_errors(trajectories)
    obj = obj_value(errors, A, f_true, bF=0)
    return obj


def obj_vec(vec):
    A, f = devectorize(vec)
    return obj(A, f)


def build_rhs(con):
    rhs = []
    for k in range(num_ivs):
        rhs.append(con[int(f_class[k])])
    return np.array(rhs)


def build_rhsFromLearn(con):
    rhs = []
    for k in range(num_ivs):
        rhs.append(con[int(learn_class[k])])
    return np.array(rhs)


def assemble_errors(trajectories):
    errors = []
    for trajectory, target, indices in zip(trajectories, targets, eval_indices):
        y_j = trajectory[indices]
        target = np.asarray(target)
        errors.append(y_j-target)
    return errors

# Raw data trajectory plot routine
def plot_trajectories(**kwargs):
    fig = plt.figure()
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    for i in plot_indices:
        target = targets[i]
        plt.plot(*np.array(target).T)
        plt.plot(*target[0], marker='o', color='k', alpha=0.5)
        plt.plot(*target[-1], marker='x', color='r')
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    plt.savefig("results/data_trajectories.pdf")
    if 'block' in kwargs:
        plt.show(block=kwargs['block'])
    else:
        plt.show()


# Plot trajectories from A and intercepts
def plot_trajectoriesA(A, rhs, **kwargs):
    fig = plt.figure()
    plt.xlim(-6,6)
    plt.ylim(-6,6)

    # Preparation to draw eigenvectors of A
    evs = np.real(np.linalg.eig(A)[1])
    ews = np.abs(np.linalg.eig(A)[0])
    ews = ews/np.min(ews)
    evs = (ews * evs).T
    evs_ts = np.array([[t*vec for t in np.linspace(-2,2,100)] for vec in evs])

    if 'noise' in kwargs:
        if kwargs['noise']:
            trajectories = forward_trajectoriesNoise(A, rhs)
    else:
        trajectories = forward_trajectories(A, rhs)
        
    for i in plot_indices:
        plt.plot(trajectories[i].T[0],trajectories[i].T[1])

    if 'x' in kwargs:
        if kwargs['x']:
            for i in plot_indices:
                x = np.linalg.solve(A, -rhs[i])
                if 'evs' in kwargs:
                    if kwargs['evs']:
                        for vec in evs_ts:
                            plt.plot(vec.T[0]+x[0], vec.T[1]+x[1], color='r', linestyle='dashed', linewidth=1)
                plt.plot(*x, marker='x', color='k')

    # Labels corresponding to the concrete panel study setting
    plt.xlabel('Home')
    plt.ylabel('Work')
    
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'name' in kwargs and not args.simulate:
        plt.savefig("results/trajectories_" + kwargs['name'] + ".pdf")
    if 'block' in kwargs:
        plt.show(block=kwargs['block'])
    else:
        plt.show()


# Plot noisy SDE simulation trajectories from simulation
def plot_simSDEtrajectories(**kwargs):
    fig = plt.figure()
    plt.xlim(-6,6)
    plt.ylim(-6,6)
        
    for i in plot_indices:
        plt.plot(trajectories_sim[i].T[0],trajectories_sim[i].T[1])

    # Labels corresponding to the concrete panel study setting
    plt.xlabel('Home')
    plt.ylabel('Work')
    
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'block' in kwargs:
        plt.show(block=kwargs['block'])
    else:
        plt.show()
    

# Routine to create time evolution plots for each matrix exponential parameter
def regression_evolution(mat, name=""):
    ex_mat = []
    for t in times:
        ex_mat.append(expm(t*mat))

    ex_mat = np.array(ex_mat).reshape(len(times), M, M)

    def plot_cross(tens,text="Unstandardised crossregression", block_it=False):
        fig = plt.figure()
        plt.xlim(0, times[-1])
        plt.ylim(-.1, 1)
        plt.plot(times,tens[:, 1, 0], label=r"satisfaction with health $\rightarrow$ satisfaction with work")
        plt.plot(times,tens[:, 0, 1], label=r"satisfaction with work $\rightarrow$ satisfaction with health")
        plt.legend()
        plt.title(text + name)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.savefig("results/crossregress_" + name.strip(" ()") + ".pdf")
        plt.show(block=block_it)

    def plot_auto(tens,text="Autoregression", block_it=False):
        fig = plt.figure()
        plt.xlim(0, times[-1])
        plt.ylim(-.1, 1)
        plt.plot(times,tens[:, 0, 0], label="satisfaction with health")
        plt.plot(times,tens[:, 1, 1], label="satisfaction with work")
        plt.legend()
        plt.title(text + name)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.savefig("results/autoregress_" + name.strip(" ()") + ".pdf")
        plt.show(block=block_it)

    plot_cross(ex_mat)
    plot_auto(ex_mat)
