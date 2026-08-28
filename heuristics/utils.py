import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from scipy.optimize import newton
from scipy.optimize import root_scalar
import random
import os
import csv
import time

def objective(x, A, w):
    """Objective function to maximize."""
    # Calculate the weighted sum for each row
    row_sums = np.dot(A, w * x)  # Element-wise multiplication of w and x
    # Add a small epsilon to avoid log(0)
    epsilon = 0.0
    return -np.sum(np.log(row_sums + epsilon))

def optimize_allocation(A, w, k, cap, eps, x0):
    """
    Solves the optimization problem.
    
    Parameters:
        A: numpy.ndarray, shape (n, m), non-negative matrix
        w: numpy.ndarray, shape (m,), weight vector (currently unused in the objective)
        k: float, positive constant
        cap: numpy.ndarray, shape (m,), capacity vector

    Returns:
        result: scipy.optimize.OptimizeResult, contains the solution and status
    """
    n, m = A.shape

    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: k - np.sum(x)},  # \sum x_j <= k
        {'type': 'ineq', 'fun': lambda x: cap - x},       # x_j <= c_j
        {'type': 'ineq', 'fun': lambda x: x}             # x_j >= 0
    ]

    # Bounds for each variable (0 <= x_j <= c_j)
    bounds = [(1e-7, c) for c in cap]

    # Optimize
    result = minimize(
        objective,
        x0,
        args=(A, w),
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 20, 'ftol': eps}
    )

    return result

def round_preserving_sum(x):
    k = int(round(np.sum(x)))
    floor_x = np.floor(x)
    decimal_part = x - floor_x
    num_ceil = int(k - np.sum(floor_x))
    indices = np.argsort(-decimal_part)[:num_ceil]
    y = floor_x.copy()
    y[indices] += 1
    return y.astype(int)


def frac_core(k, A, cap):
    n,m = A.shape
    if cap is None:
        cap = np.ones(m)
    w = np.ones(m)
    eps = 1.0
    epssum = 1.0
    x0 = np.minimum(cap, k / m)
    while True:
        # Solve optimization problem
        result = optimize_allocation(A, w, k, cap, 1e-4*np.minimum(eps,1.0), x0)
        #result = optimize_allocation(A, w, k, cap, 1e-9, x0)
        # Output the result
        if result.success:
            x = result.x
            x0 = x.copy()
            obj = -result.fun
        else:
            #print(result.x)
            x = result.x
            x0 = x.copy()
            #print("Optimization failed:", result.message)
            
        partial_f_partial_x = np.zeros_like(x)
        for j in range(len(x)):
            xfj = x.copy()
            xfj[j] = cap[j]
            wfj = w.copy()
            #wfj[j] = 1.0
            wx = wfj * xfj
            wu = A @ wx
            if np.any(wu == 0):
                raise ValueError("wu contains zeros, cannot compute log or divide by zero.")
            partial_f_partial_x[j] = np.sum((A[:, j]) / wu)
        # Outputs
        new_w = (n/k) / partial_f_partial_x
        eps = np.linalg.norm(w - np.minimum( new_w, 1.0 ))
        epssum = epssum * 0.9 + 0.1*eps*eps
        print('current epsilon:', eps)
        if eps<1e-3:
            break
        #w = w + (new_w - w) * (0.1 / np.sqrt(epssum+1e-20) )
        w = w + (new_w - w) * 0.3
        w = np.minimum( w , 1.0 )
        #print(eps)
    return w,x


def update_margins(A,w,x,u,low_margin,high_margin):
    np.copyto(u, A @ (w*x))
    np.copyto(low_margin, w * (A.T @ (1.0/(u+1))))
    np.copyto(high_margin, w * (A.T @ (1.0/(u+1e-10))))
def disc_core(k, A, cap, w=None, x=None):
    A = np.round(A).astype(int)
    k = round(k)
    n,m = A.shape
    if cap is None:
        cap = np.ones(m, dtype=int)
    else:
        cap = np.round(cap).astype(int)
    #
    if w is None:
        w = np.ones(m)
    if x is None:
        x = np.zeros(m, dtype=int)
        lft = k
        for i in range(m):
            if lft>=cap[i]:
                x[i]=cap[i]
                lft-=cap[i]
            else:
                x[i]=lft
                break
    else:
        x = round_preserving_sum(x)
    
    u = A @ (w*x)
    low_margin = A.T @ (1.0/(u+1))
    high_margin = A.T @ (1.0/(u+1e-10))
    eps = 1e-1
    cnt = 0
    while np.max(low_margin)>n/k:
        cnt += 1
        if eps>1e-6 and cnt*eps>1:
            eps*=0.5
        """
        if cnt == 1 or cnt == 100 or cnt%50000==0:
            print(cnt, n/k, np.max(low_margin))
            print(low_margin)
            print(w)
            print(x)
            print(cap)
        """
        inc_c = np.argmax(low_margin)
        if x[inc_c]<cap[inc_c]:
            x[inc_c] += 1
            #print('increasing x', inc_c)
            update_margins(A,w,x,u,low_margin,high_margin)
            while True:
                valid_indices = np.where(x > 0)[0]
                dec_c = valid_indices[np.argmin(high_margin[valid_indices])]
                if w[dec_c]>1.0-5e-7:
                    x[dec_c]-=1
                    #print('decreasing x', dec_c)
                    update_margins(A,w,x,u,low_margin,high_margin)
                    break
                w[dec_c]+=eps
                w[dec_c] = min(1.0, w[dec_c])
                #print('increasing w', dec_c)
                update_margins(A,w,x,u,low_margin,high_margin)
        else:
            w[inc_c] -= eps
            #print('decreasing w', inc_c)
            update_margins(A,w,x,u,low_margin,high_margin)
    print('finish in', cnt, 'iterations')
    return w,x
