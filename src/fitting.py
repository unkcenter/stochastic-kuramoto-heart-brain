import numpy as np
from src.dynamics import run_stochastic_kuramoto

def calculate_loss(K_HB, sigma, R_target, H_target, N, T, dt, omega_B, omega_H, K_B, runs=3):
    """
    Calculates the normalized quadratic loss between simulated values and real-world targets.
    """
    R_sims = []
    H_sims = []
    
    for _ in range(runs):
        _, R_history, entropy_history, _, _, _ = run_stochastic_kuramoto(
            N, T, dt, omega_B, omega_H, K_B, K_HB, sigma
        )
        # Analyze only the steady state (last 50%)
        steady_state_start = len(R_history) // 2
        R_sims.append(np.mean(R_history[steady_state_start:]))
        H_sims.append(np.mean(entropy_history[steady_state_start:]))
        
    err_R = ((np.mean(R_sims) - R_target) / R_target) ** 2
    err_H = ((np.mean(H_sims) - H_target) / H_target) ** 2
    
    return err_R + err_H

def run_grid_search(R_target, H_target, N, T, dt, omega_B, omega_H, K_B, k_hb_grid, sigma_grid, runs_per_k=3):
    """
    Runs a 2D grid search over K_HB and sigma to find the optimal biological parameters.
    """
    loss_matrix = np.zeros((len(k_hb_grid), len(sigma_grid)))
    
    for i, K_HB in enumerate(k_hb_grid):
        for j, sigma in enumerate(sigma_grid):
            loss_matrix[i, j] = calculate_loss(
                K_HB, sigma, R_target, H_target, N, T, dt, omega_B, omega_H, K_B, runs=runs_per_k
            )
            
    best_idx = np.unravel_index(np.argmin(loss_matrix), loss_matrix.shape)
    best_K_HB = k_hb_grid[best_idx[0]]
    best_sigma = sigma_grid[best_idx[1]]
    
    return best_K_HB, best_sigma, loss_matrix
