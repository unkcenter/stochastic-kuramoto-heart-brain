import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fitting import run_grid_search

def main():
    # 1. DEFINE TARGET EXPERIMENTAL DATA (MOCK PATIENT)
    R_experimental = 0.65       # Mean target brain-heart phase locking value
    H_experimental = 2.20       # Mean target phase Shannon entropy (in nats)

    # 2. DEFINE SYSTEM CONFIGURATIONS
    N = 100                     # Number of neural oscillators
    T = 25.0                    # Simulation duration (s)
    dt = 0.05                   # Time step (s)

    # Autonomic and neural frequency parameters
    f_H = 0.10                  # 0.1 Hz cardiac resonant frequency
    omega_H = 2 * np.pi * f_H
    
    np.random.seed(42)
    frequencies_brain = np.random.normal(loc=0.12, scale=0.04, size=N)
    omega_B = 2 * np.pi * frequencies_brain
    
    K_B = 0.15                  # Endogenous brain coupling strength

    # 3. CONFIGURE PARAMETER GRID FOR ESTIMATION
    grid_size = 15
    k_hb_grid = np.linspace(0.1, 0.8, grid_size)
    sigma_grid = np.linspace(0.1, 0.6, grid_size)

    print("Starting stochastic parameter optimization against target patient metrics...")
    
    # 4. EXECUTE PARAMETER FITTING
    best_K_HB, best_sigma, loss_matrix = run_grid_search(
        R_target=R_experimental,
        H_target=H_experimental,
        N=N,
        T=T,
        dt=dt,
        omega_B=omega_B,
        omega_H=omega_H,
        K_B=K_B,
        k_hb_grid=k_hb_grid,
        sigma_grid=sigma_grid,
        runs_per_k=3
    )

    # 5. PRINT INVERSE PROBLEM ESTIMATION RESULTS
    print("\n" + "="*50)
    print("             ESTIMATED BIOLOGICAL PARAMETERS")
    print("="*50)
    print(f"Optimal Vagal/Baroreflex Coupling (K_HB): {best_K_HB:.3f}")
    print(f"Endogenous Cortical Noise Intensity (sigma):  {best_sigma:.3f}")
    print(f"Minimum Parameter Estimation Loss:            {np.min(loss_matrix):.5f}")
    print("="*50)

    # 6. PLOT LOSS LANDSCAPE CONTOUR
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(sigma_grid, k_hb_grid, loss_matrix, levels=20, cmap='viridis_r')
    plt.colorbar(contour, label="Normalized Quadratic Loss (Fitting Error)")
    plt.plot(best_sigma, best_K_HB, 'r*', markersize=15, 
             label=f"Patient Best Fit\n(K_HB={best_K_HB:.2f}, σ={best_sigma:.2f})")
    
    plt.title("Parameter Fitting: Kuramoto Model vs. Target EEG/ECG Data", fontsize=12)
    plt.ylabel("Heart-to-Brain Coupling Strength (K_HB)")
    plt.xlabel("Cortical Noise Intensity (σ)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the output figure
    output_dir = os.path.join(os.path.dirname(__file__), '../notebooks')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'loss_landscape.png'), dpi=300)
    print("\nFitting visualization saved successfully in 'notebooks/loss_landscape.png'.")
    plt.show()

if __name__ == "__main__":
    main()
