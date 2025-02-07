 
# Exotic Derivative Pricing: Asian & Barrier Options

https://github.com/user-attachments/assets/0bf6073d-b6a4-4ce1-a9c2-f4e2540de9d4


Leveraging Monte Carlo simulation and Euler discretization of Geometric Brownian Motion (GBM), this project prices digital and barrier options, computes option payoffs, and visualizes simulated asset paths and price distributions through informative graphs. An interactive Streamlit app (located in `app.py`) provides a user-friendly interface to explore the pricing models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Simulation](#running-the-simulation)
  - [Interactive Streamlit App](#interactive-streamlit-app)
- [Theory & Methodology](#theory--methodology)
- [Results & Visualization](#results--visualization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This repository implements an exotic derivative pricing framework focused on Asian and barrier options. Using a Monte Carlo approach and the Euler discretization method to simulate GBM paths, the project:
- Prices digital (cash-or-nothing) options.
- Prices barrier options (up‑and‑out for calls and corresponding variants for puts).
- Computes and compares option payoffs.
- Visualizes the underlying asset paths, terminal price histograms, and payoff functions.

An interactive [Streamlit](https://streamlit.io/) app (in `app.py`) allows users to adjust parameters and immediately see the impact on option prices and simulated graphs.

## Features

- **Monte Carlo Simulation:** Generate asset price paths using Euler discretization of GBM.
- **Digital Option Pricing:** Compute the discounted payoff for digital options.
- **Barrier Option Pricing:** Incorporate barrier conditions (e.g., up-and‑out for calls, down‑and‑out for puts) to determine option validity.
- **Visualization:** Plot sample asset paths, histograms of terminal prices, and digital payoff functions with barrier levels clearly marked.
- **Interactive Streamlit App:** An easy-to-use interface for real‑time parameter adjustment and visualization.

## Project Structure


Exotic-Derivative-Pricing-Asian-Barrier-
├── Asian notbook.ipynb         # Jupyter Notebook demonstrating Asian option pricing
├── Asian..py                   # Python script for Asian option pricing routines
├── BarrierOption.py            # Python module for barrier option pricing
├── DigitalPayoffs_CostReduction.ipynb  # Notebook showcasing digital payoff calculations and cost reduction analysis
├── app.py                      # Streamlit app for interactive exploration
└── requirements.txt            # List of required packages


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/shubh123a3/Exotic-Derivative-Pricing-Asian-Barrier-.git
   cd Exotic-Derivative-Pricing-Asian-Barrier-
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Simulation

You can run the core simulation scripts (for example, the barrier and digital option pricing routines) directly using Python. For instance:

```bash
python BarrierOption.py
```

This will:
- Generate Monte Carlo asset paths using Euler discretization.
- Compute option payoffs for digital and barrier options.
- Plot asset paths, histograms, and payoff functions.
- Print option prices and related statistics to the console.

### Interactive Streamlit App

For an interactive exploration of the model:
1. Ensure you have installed Streamlit (it is included in `requirements.txt`).
2. Run the app using:

   ```bash
   streamlit run app.py
   ```

The app allows you to:
- Modify simulation parameters (initial asset price, volatility, interest rate, barrier level, etc.).
- Visualize updated asset path simulations and payoff graphs in real time.
- Compare digital option pricing against barrier option pricing interactively.

## Theory & Methodology





### Geometric Brownian Motion & Euler Discretization

The asset price \( S(t) \) is modeled via GBM:
\[
dS(t) = r\, S(t)\, dt + \sigma\, S(t)\, dW(t)
\]
Using Euler discretization with time-step \( \Delta t \):
\[
S_{t+\Delta t} = S_t + r\, S_t\, \Delta t + \sigma\, S_t\, \sqrt{\Delta t}\, Z
\]
where \( Z \sim \mathcal{N}(0,1) \).

### Option Payoff & Pricing

- **Digital Option:**  
  The discounted payoff is computed as:
  \[
  V_0 = e^{-rT} \, \mathbb{E}[\text{Payoff}(S_T)]
  \]
- **Barrier Option:**  
  For an up‑and‑out call, if the asset price exceeds the barrier \( S_u \) during the simulation, the option payoff is set to zero. The pricing then becomes:
  \[
  V_0 = e^{-rT} \, \mathbb{E}[\text{Payoff}(S_T) \times \mathbb{I}(\text{barrier not breached})]
  \]

Monte Carlo simulations are used to estimate these expectations by averaging over many simulated paths.

## Results & Visualization

The project produces several informative graphs:
- **Digital Payoff Curve:** Displays the payoff function (e.g., \(\max(S-K, 0)\)) with the barrier level marked.
- **Asset Path Simulations:** Shows multiple sample paths alongside the mean path and a horizontal line indicating the barrier level.
- **Histograms:** Compare the distribution of terminal asset prices and Asian average prices.

These visualizations help validate the pricing model and offer insights into the impact of barrier conditions on option payoffs.

## Contributing

Contributions, suggestions, and bug reports are welcome. Feel free to open an issue or submit a pull request. When contributing:
- Follow the existing code style.
- Update the README with any changes to features or usage.
- Write tests for new features if applicable.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Inspired by quantitative finance research and textbooks (e.g., Hull, John C. "Options, Futures, and Other Derivatives").
- Special thanks to the open-source community for providing resources and inspiration in developing advanced option pricing models.
- Developed by [shubh123a3](https://github.com/shubh123a3).

```

---

This README provides a comprehensive overview of the project's objectives, methods, and usage instructions. Customize sections as needed to match any additional details or changes in your code.
