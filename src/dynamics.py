import numpy as np

def run_stochastic_kuramoto(N, T, dt, omega_B, omega_H, K_B, K_HB, sigma_noise):
    """
    Simulates a network of N coupled brain oscillators driven by a cardiac pacemaker
    using Euler-Maruyama integration under additive Gaussian white noise.
    """
    steps = int(T / dt)
    time = np.linspace(0, T, steps)
    theta = np.random.uniform(-np.pi, np.pi, N)
    
    R_history = np.zeros(steps)
    psi_history = np.zeros(steps)
    entropy_history = np.zeros(steps)
    theta_H_history = np.zeros(steps)
    
    def calculate_shannon_entropy(phases, num_bins=18):
        wrapped_phases = np.mod(phases, 2 * np.pi)
        counts, _ = np.histogram(wrapped_phases, bins=num_bins, range=(0, 2 * np.pi))
        probs = counts / len(phases)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    for step in range(steps):
        t = step * dt
        theta_H = omega_H * t
        
        # Calculate mean field order parameter
        z = np.mean(np.exp(1j * theta))
        R = np.abs(z)
        psi = np.angle(z)
        
        R_history[step] = R
        psi_history[step] = psi
        theta_H_history[step] = theta_H
        entropy_history[step] = calculate_shannon_entropy(theta)
        
        # SDE update: dtheta_i = [omega_i + K_B*R*sin(psi - theta_i) + K_HB*sin(theta_H - theta_i)] * dt + sigma * dW_i
        noise = np.random.normal(0, 1, N)
        dtheta = (omega_B + K_B * R * np.sin(psi - theta) + K_HB * np.sin(theta_H - theta)) * dt \
                 + sigma_noise * np.sqrt(dt) * noise
                 
        theta += dtheta
        
    return time, R_history, entropy_history, psi_history, theta_H_history, theta
