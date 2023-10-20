#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hannesmeinlschmidt
"""

from VarCT_optimize import *
import numdifftools as nd

def check_grad():
    # Determine random matrix A und intercepts f in which the gradient is checked,
    # also random normalized directions H and h_f
    scale = 1
    A = 2 * scale * np.random.random_sample((M, M)) - scale
    while not np.all(np.real(np.linalg.eig(A)[0] <= 0)):
        A = 2 * scale * np.random.random_sample((M, M)) - scale
  
    H = 2 * np.random.random_sample((M, M)) - 1
    H = H / np.linalg.norm(H)

    if learn_intercepts:
        f = 2 * scale * np.random.random_sample((learn_num, M)) - scale
        h_f = 2 * np.random.random_sample((learn_num, M)) - 1
        h_f = h_f / np.linalg.norm(h_f)
    else:
        f = f_true
        h_f = np.zeros((learn_num, M))


    # Calculate central different quotients in directions H and h_f up to order 9
    diffqs = []
    order_start = 3
      
    for k in range(10)[order_start:]:
        A_H = A + (10 ** (-k)) * H
        A_HMinus = A - (10 ** (-k)) * H
        f_h = f + (10 ** (-k)) * h_f
        f_hMinus = f - (10 ** (-k)) * h_f
        obj_dir = obj(A_H, f_h)
        obj_dirMinus = obj(A_HMinus, f_hMinus)
        diffq = (obj_dir - obj_dirMinus) * (10 ** k) / 2
        diffqs.append(diffq)

    grads = []

    # Calculate gradient by possibly various methods to compare
    # Gradients collected in list "grads"
    # Associated identifiers in list "methods"
    
    # Adjoint method as used in the code
    methods = ["adjoint method"]
    start = time.time()
    grad_Af = gradient(A, f)
    if learn_intercepts:
        grads.append(grad_Af)
    else:
        grads.append(grad_Af[0])
    end = time.time()
    print("Computation time adjoint method: {:g}s".format(end-start))

    # The following gradients are only tested without intercept learning
    if not learn_intercepts:
        # Calculate full differential numerically
        # This is in general a bad idea
        # but if M and the number of intercept classes to learn is small, it can be fine
        methods.append("full differential")
        start = time.time()
        # Set up all directions
        H_dirs = []
        for k in range(M*M):
            Hk = np.zeros(M*M)
            Hk[k] = 1
            Hk = Hk.reshape((M, M))
            H_dirs.append(Hk)

        grad_dq = np.zeros((M, M))
        for H_dir in H_dirs:
            H_dir = H_dir.reshape(M, M)
            k = 6
            A_Hdir = A + (10 ** (-k)) * H_dir
            A_mHdir = A - (10 ** (-k)) * H_dir
            obj_AHdir = obj_onlyA(A_Hdir)
            obj_AmHdir = obj_onlyA(A_mHdir)
            diffq = (obj_AHdir - obj_AmHdir) * (10 ** k) / 2
            grad_dq += diffq * H_dir

        grads.append(np.asarray(grad_dq))
        end = time.time()
        print("Computation time full differential: {:g}s".format(end-start))

        #### NumDiff
        methods.append("NumDiff")
        start = time.time()
        A = np.asarray(A.reshape(M*M, 1))
        grad_nd = nd.Gradient(obj_onlyA)
        grad_A = grad_nd(A)
        grads.append(np.asarray(grad_A))
        end = time.time()
        print("Computation time numdiff: {:g}s".format(end-start))

    for k, grad_A in enumerate(grads):
        print("--------------")
        print("Comparing " + str(methods[k]) + " directional derivative to differential quotient...")
        A = A.reshape(M, M)
        H_vec = H.reshape(1, M*M)
        if learn_intercepts:
            gradA_vec = grad_A[0].reshape(1, M*M)
        else:
            gradA_vec = grad_A.reshape(1, M*M)

        dir_deriv = np.squeeze(np.inner(gradA_vec, H_vec))

        if learn_intercepts:
            for grad_j, h_j in zip(grad_A[1], h_f):
                dir_deriv += np.inner(grad_j, h_j)
     
        print("Directional derivative: {:g}".format(dir_deriv))
        for j, diffq in enumerate(diffqs):
            relerror = np.absolute(diffq-dir_deriv)/np.absolute(diffq)
            print("Differential quotient order {}, relative error {:g}".format(j+order_start, relerror))

      
if __name__ == "__main__":
    check_grad()
