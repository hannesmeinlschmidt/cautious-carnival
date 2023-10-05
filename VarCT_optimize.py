#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""

from scipy.optimize import minimize
from scipy import integrate

from VarCT_startup import *
from VarCT_functions import *


# Build gradient of objective function
def gradient(A, f):
    A = A.reshape(M,M)
    rhs = build_rhsFromLearn(f)
    trajectories = forward_trajectories(A, rhs)
    fvs = assemble_errors(trajectories)
    
    def adj_fA(x, t):
        return adj_f(x, t, A)

    trajectories_bw = backward_trajectories(fvs, adj_fA)
    gradient_A = np.zeros((M, M), dtype=float)
    gradient_f_all = []
    # j loop
    # Iterates subjects (backwards trajectories start at each target)
    for j, trajectory_bw_j in enumerate(trajectories_bw):
        gradient_Aj = np.zeros((M, M), dtype=float)
        gradient_fj = np.zeros(M, dtype=float)
        # i loop
        # Iterates backwards trajectories per subject
        for i, trajectory_bw_ij in enumerate(trajectory_bw_j):
            integrand = []
            # Builds a time-continuous matrix function and integrates
            times_ij = times_iv[j][0:eval_indices[j][i]+1]
            for k in range(len(times_ij)):
                matrix = np.dot(trajectory_bw_ij[k].reshape(M, 1),
                                trajectories[j][k].reshape(1, M))
                integrand.append(matrix)
            gradient_Aj += integrate.simpson(integrand, x=times_ij, axis=0)
            gradient_fj += integrate.simpson(trajectory_bw_ij, x=times_ij, axis=0)
        gradient_f_all.append((obj_div[j]**(-1)) * gradient_fj)
        gradient_A += (obj_div[j]**(-1)) * gradient_Aj
    gradient_A += beta_A * A
    gradient_A = np.squeeze(gradient_A.reshape(M*M,1))

    # Yet another detour due to intercept classes learning
    gradient_f = np.zeros((learn_num, M))
    for j, grad_fj in enumerate(gradient_f_all):
        gradient_f[int(learn_class[j])] += grad_fj

    gradient_f += beta_F * np.array(f)
    return [gradient_A, gradient_f]


# Wrapper for vector-representation of gradient
def gradient_vector(vec):
    A, f = devectorize(vec)
    grad_A, grad_f = gradient(A,f)
    grad_vec = np.array(grad_A).reshape(M*M,1)
    for fj in grad_f:
        grad_vec = np.append(grad_vec,fj)
    return grad_vec


# Optimization routine using BFGS
# The optimization routine supplied by scipy works on vectors so everything is vectorized and wrapped
def optimize_BFGS():
    # Initial optimization iterate for matrix
    A_start = A_CTSEM.reshape(M*M)
    if args.simulate:
        A_start = np.array([-1, 0, 0, -1], dtype=float)

    vec_start = A_start

    # If intercepts are learned, they are part of the optimization variable
    if learn_intercepts:
        vec_start = np.append(vec_start, np.zeros((learn_num, M)))

    # Wrapper for objective function to be minimized and its gradient
    # Depending on whether intercepts are learned or not
    def obj_local(vec):
        if learn_intercepts:
            return obj_vec(vec)
        else:
            return obj(vec, f_true)

    def grad_local(vec):
        if learn_intercepts:
            return gradient_vector(vec)
        else:
            return gradient(vec, f_true)[0]

    print("Gradient via adjoint calculus")
    print("")
    res = minimize(obj_local, vec_start, method='BFGS', jac=grad_local, options={'disp': True})

    # Unwrap vectorization        
    if learn_intercepts:
        vec_guess = res.x
    else:
        vec_guess = np.append(res.x, f_true)
    return devectorize(vec_guess)


# Main routine and comparison with CTSEM matrix
def optimize():
    if args.simulate:
        rhs_true = build_rhs(f_true)
        plot_simSDEtrajectories(block=False, title="Simulated trajectories (SDE)")
    else:
        rhs_true = np.zeros((num_ivs, M))
        plot_trajectoriesA(A_CTSEM, rhs_true, x=True, block=False, title="ctsem matrix trajectories", evs=True, name="ctsem")
        plot_trajectories(title="Trajectories interpolated from data", block=False)
        regression_evolution(A_CTSEM, name=" (ctsem)")

    print("---------- OPTIMIZATION ----------")
    start = time.time()
    A_guess, f_guess = optimize_BFGS()
    end = time.time()

    rhs_guess = build_rhsFromLearn(f_guess)
  
    print("")
    print("Computation time: {:1.2f}s".format(end-start))
    print("-------- OPTIMIZATION DONE --------")
    print("")
    print("------------- RESULT --------------")
    print("Matrix:")
    matprint(A_guess)
    print("Eigenvalues:\n{}".format(np.linalg.eig(A_guess)[0]))
    print("Eigenvectors (columns):\n{}".format(np.linalg.eig(A_guess)[1]))
    objVal_Aguess = obj(A_guess, f_guess)
    print("Objective function: {:g}".format(objVal_Aguess))
    print("")
    print("------------ REFERENCE ------------")

    if args.simulate:
        A_compare = A_true
        print("Simulated matrix:")
    else:
        A_compare = A_CTSEM
        print("ctsem matrix:")

    matprint(A_compare)
    print("Eigenvalues:\n{}".format(np.linalg.eig(A_compare)[0]))
    print("Eigenvectors (columns):\n{}".format(np.linalg.eig(A_compare)[1]))
    objVal_Acomp = obj_onlyA(A_compare)
    print("Objective function: {:g}".format(objVal_Acomp))
    print("")
    print("Relative error in matrix: {:.2%}".format(np.linalg.norm(A_guess-A_compare)/np.linalg.norm(A_compare)))
    print("Relative change in objective: {:.2%}".format((objVal_Acomp-objVal_Aguess)/objVal_Acomp))
    if learn_intercepts:
        if learn_num == 1:
            print("Resulting (average) Target:")
            f_av = np.average(rhs_guess, axis=0)
            print(np.linalg.solve(A_guess, -f_av))
        if args.simulate:
            rhs_true = build_rhs(f_true)
            print("Relative error in f: {:.2%}".format(np.linalg.norm(rhs_guess-rhs_true)/np.linalg.norm(rhs_true)))
    
    print("")

    if not args.simulate:
        regression_evolution(A_guess, name=" (learning)")
    plot_trajectoriesA(A_guess, rhs_guess, x=True, title="Learned trajectories", name="learned")

    return A_guess, f_guess


if __name__ == "__main__":
    A_guess, f_guess = optimize()
