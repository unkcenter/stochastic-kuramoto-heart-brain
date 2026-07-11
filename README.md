# Stochastic Kuramoto Heart-Brain Coherence Model

A mathematically rigorous, open-source computational neuroscience framework designed to simulate and study phase-locking and entrainment between the cardiovascular system (vagal/baroreflex pace) and cortical neural oscillators. 

Using a **Stochastic Kuramoto Model under asymmetric periodic driving**, this project models how breathing at the resonant frequency (~0.1 Hz) reduces overall cortical phase entropy, facilitating states of high cognitive suggestibility, neuroplasticity, and focused awareness. It also resolves the **inverse problem (parameter estimation)** to fit simulated states to target real-world human EEG/ECG data.

---

## Mathematical Formulation

### 1. Stochastic Differential Equation (SDE)
We model the phase dynamics of $N$ cortical oscillators subjected to an asymmetric periodic heart driver and additive Gaussian white noise (integrated via Euler-Maruyama):

$$d\theta_i(t) = \left[ \omega_i + \frac{K_B}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i) + K_{HB} \sin(\theta_H - \theta_i) \right] dt + \sigma dW_i(t)$$

Where:
- $\theta_i(t)$ is the phase of the $i$-th cortical oscillator.
- $\omega_i$ represents the natural intrinsic frequency of local cortical fluctuations.
- $K_B$ is the internal coupling strength of the brain network.
- $\theta_H = \omega_H t$ is the phase of the cardiac pacemaking driver (where $f_H \approx 0.1\text{ Hz}$).
- $K_{HB}$ represents the heart-to-brain coupling strength (via vagal/baroreceptive afferents).
- $\sigma$ represents endogenous brain noise (stochastic synaptic firing).
- $dW_i(t)$ represents independent Wiener processes (Brownian motion).

### 2. Information Entropy Reduction
During incoherent states ($K_{HB} \to 0$), phase distribution is uniform, yielding maximum Shannon Entropy $H_{\text{max}} = \ln(M)$.

Under cardiac driving above the critical coupling threshold ($K_{HB} > K_c \approx 0.25$), the brain transitions into a coherent attractor. The phase density $p(\theta)$ is modeled by a **Von Mises distribution**:

$$p(\theta) = \frac{e^{r \cos(\theta - \psi)}}{2\pi I_0(r)}$$

The corresponding Shannon Entropy is drastically reduced:

$$H(r) = \ln(2\pi I_0(r)) - r \frac{I_1(r)}{I_0(r)}$$

This drop in entropy ($\Delta H$) represents the mathematical explanation for the collapse of top-down mental constraints, enabling subconscious restructuring.

---

## Repository Structure

```text
stochastic-kuramoto-heart-brain/
│
├── .gitignore              # Files ignored by Git
├── LICENSE                 # MIT License
├── README.md               # Main project documentation
├── requirements.txt        # Python dependencies
│
├── src/                    # Core mathematical package
│   ├── __init__.py         # Package entrypoint
│   ├── dynamics.py         # SDE Euler-Maruyama Kuramoto integration
│   └── fitting.py          # Inverse parameter fitting (Grid Search)
│
└── examples/               # Computational experiment runs
    └── run_fitting_demo.py # Parameter estimation and visualization script
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/unkcenter/stochastic-kuramoto-heart-brain.git
   cd stochastic-kuramoto-heart-brain
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+ installed. Then, install requirements:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start (Running the Demo)

To run the parameter estimation simulation, which fits the Kuramoto model to mock target patient data ($R = 0.65, H = 2.20$) and generates a Loss Landscape contour map:

```bash
python examples/run_fitting_demo.py
```

### Expected Output:
- The terminal will display the progress of the grid search sweep over $K_{HB}$ and $\sigma$.
- It will print the optimal estimated parameters found for the patient.
- It will generate and display a contour plot of the Loss Landscape (saving it automatically to `notebooks/loss_landscape.png`).

---

## Parameter Estimation Map (Sample Result)
The optimization algorithm maps experimental targets onto a continuous, convex loss landscape, localizing the global minimum of error (indicated by the red star). This allows researchers to estimate individual biological parameters such as heart-brain connectivity ($K_{HB}$) and cortical noise ($\sigma$) from human EEG/ECG.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
