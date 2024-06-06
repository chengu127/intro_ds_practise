#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:54:05 2024

@author: chengu
"""

def is_irreducible(P):
    n = len(P)
    reachability = np.linalg.matrix_power(P, n - 1)
    return np.all(reachability > 0)

irreducible = is_irreducible(P)
print(f"Is the Markov chain irreducible? {'Yes' if irreducible else 'No'}")

# Part 3: Check if the Markov chain is irreducible
def is_irreducible(P):
    n = len(P)
    reachability = np.linalg.matrix_power(P, n - 1)
    return np.all(reachability > 0)

irreducible = is_irreducible(P)
print(f"Is the Markov chain irreducible? {'Yes' if irreducible else 'No'}")

# Part 4: Stationary distribution
A = np.transpose(P) - np.eye(3)
A[-1, :] = 1
b = np.zeros(3)
b[-1] = 1
stationary_distribution = solve(A, b)
print(f"Stationary distribution: {stationary_distribution}")

# 1. Calculate the stationary distribution
def get_stationary_distribution(P):
    A = np.transpose(P) - np.eye(len(P))
    A[-1, :] = 1
    b = np.zeros(len(P))
    b[-1] = 1
    stationary_distribution = solve(A, b)
    return stationary_distribution

stationary_distribution = get_stationary_distribution(P)
print(f"Stationary distribution: {stationary_distribution}")

# 2. Check reversibility
def is_reversible(P, stationary_distribution):
    n = len(P)
    for i in range(n):
        for j in range(n):
            if stationary_distribution[i] * P[i, j] != stationary_distribution[j] * P[j, i]:
                return False
    return True

reversible = is_reversible(P, stationary_distribution)
print(f"Is the Markov chain reversible? {'Yes' if reversible else 'No'}")

# 3. Check aperiodicity
def is_aperiodic(P):
    n = len(P)
    for i in range(n):
        if P[i, i] > 0:
            return True

    # Alternative method: Check GCD of cycles
    gcd = np.gcd.reduce([np.gcd.reduce([np.count_nonzero(P[:,i]),np.count_nonzero(P[i,:])]) for i in range(n)])
    return gcd == 1

aperiodic = is_aperiodic(P)
print(f"Is the Markov chain aperiodic? {'Yes' if aperiodic else 'No'}")


# Part 5: Expected number of steps until first entering downtown from suburbs
# Let E be the expected number of steps
E = np.zeros(3)
tolerance = 1e-6
max_steps = 100

for _ in range(max_steps):
    E_new = np.zeros(3)
    E_new[0] = 0  # Downtown: we are already there
    E_new[1] = 1 + P[1, 0] * E[0] + P[1, 1] * E[1] + P[1, 2] * E[2]  # Suburbs
    E_new[2] = 1 + P[2, 0] * E[0] + P[2, 1] * E[1] + P[2, 2] * E[2]  # Countryside
    
    if np.max(np.abs(E - E_new)) < tolerance:
        break
    E = E_new

expected_steps_to_downtown_from_suburbs = E[1]
print(f"Expected number of steps to first enter downtown from suburbs: {expected_steps_to_downtown_from_suburbs:.1f}")