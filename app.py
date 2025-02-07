import streamlit as st
import numpy as np
import enum
import matplotlib.pyplot as plt
from enum import Enum


# Enum for option type
class OptionType(enum.Enum):
    call = 1.0
    put = -1.0


# Common functions
def GeneratePathsGBMEuler(NoOfPaths, NoOfSteps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    S1 = np.zeros([NoOfPaths, NoOfSteps + 1])
    S1[:, 0] = S_0
    time = np.zeros([NoOfSteps + 1])
    dt = T / float(NoOfSteps)

    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.sqrt(dt) * Z[:, i]
        S1[:, i + 1] = S1[:, i] + r * S1[:, i] * dt + sigma * S1[:, i] * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt

    return {"time": time, "S": S1}


# Asian Option Functions
def asian_payoff(S, K, option_type):
    if option_type == OptionType.call:
        return np.maximum(np.mean(S, axis=1) - K, 0.0)
    else:
        return np.maximum(K - np.mean(S, axis=1), 0.0)


# Digital Barrier Functions
def UpandOutBarrier(S, T, r, K, Su, option_type):
    n_paths, n_steps = S.shape
    barrier = np.full_like(S, Su)
    hit = np.any(S > barrier, axis=1)
    discount = np.exp(-r * T)

    if option_type == OptionType.call:
        payoff = np.where(~hit, np.maximum(S[:, -1] - K, 0), 0)
    else:
        payoff = np.where(~hit, np.maximum(K - S[:, -1], 0), 0)

    return discount * np.mean(payoff)


# Streamlit App
st.title("Exotic Derivative Pricing Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Simulation Parameters")
    option_choice = st.radio("Option Type", ["Asian", "Digital Barrier"])
    S0 = st.number_input("Initial Price (S0)", value=100.0)
    r = st.number_input("Risk-free Rate (r)", value=0.05)
    T = st.number_input("Time to Maturity (T)", value=5.0)
    sigma = st.number_input("Volatility (σ)", value=0.2)
    paths = st.number_input("Number of Paths", value=5000)
    steps = st.number_input("Number of Steps", value=250)
    option_dir = st.radio("Option Direction", ["Call", "Put"])

    if option_choice == "Asian":
        K_asian = st.number_input("Strike Price (K)", value=100.0)
    else:
        K_digital = st.number_input("Strike Price (K)", value=100.0)
        Su = st.number_input("Barrier Level (Su)", value=150.0)

# Convert option direction to Enum
option_type = OptionType.call if option_dir == "Call" else OptionType.put

if option_choice == "Asian":
    # Asian Option Pricing
    paths_data = GeneratePathsGBMEuler(paths, steps, T, r, sigma, S0)
    S_paths = paths_data["S"]
    time = paths_data["time"]

    # Calculate prices
    asian_price = np.exp(-r * T) * np.mean(asian_payoff(S_paths, K_asian, option_type))
    european_price = np.exp(-r * T) * np.mean(np.maximum(S_paths[:, -1] - K_asian, 0)
                                              if option_type == OptionType.call else
                                              np.maximum(K_asian - S_paths[:, -1], 0))

    # Display results
    st.subheader("Pricing Results")
    col1, col2 = st.columns(2)
    col1.metric("European Option Price", f"{european_price:.4f}")
    col2.metric("Asian Option Price", f"{asian_price:.4f}")

    # Plots
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10))
    gs = fig1.add_gridspec(2, 2)
    # Paths plot
    ax1 = fig1.add_subplot(gs[0, :])
    for i in range(min(10, paths)):
        ax1.plot(time, S_paths[i, :], lw=0.8, alpha=0.7)
    ax1.plot(time, np.mean(S_paths, axis=0), 'k--', lw=2, label='Mean Path')
    ax1.set_title("Simulated Asset Paths")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Asset Price")

    # Terminal prices histogram
    ax2 = fig1.add_subplot(gs[1, 0])
    ax2.hist(S_paths[:, -1], bins=30, color='skyblue', edgecolor='black')
    ax2.set_title("Terminal Prices Distribution")
    ax2.set_xlabel("Price at Maturity")

    # Asian prices histogram
    ax3 = fig1.add_subplot(gs[1, 1])
    ax3.hist(np.mean(S_paths, axis=1), bins=30, color='salmon', edgecolor='black')
    ax3.set_title("Asian Average Prices Distribution")
    ax3.set_xlabel("Average Price")

    st.pyplot(fig1)

else:
    # Digital Barrier Pricing
    paths_data = GeneratePathsGBMEuler(paths, steps, T, r, sigma, S0)
    S_paths = paths_data["S"]
    S_T = S_paths[:, -1]
    time = paths_data["time"]

    # Calculate prices
    digital_price = np.exp(-r * T) * np.mean((S_paths[:, -1] > K_digital).astype(float))
    barrier_price = UpandOutBarrier(S_paths, T, r, K_digital, Su, option_type)

    # Display results
    st.subheader("Pricing Results")
    col1, col2 = st.columns(2)
    col1.metric("Digital Option Price", f"{digital_price:.4f}")
    col2.metric("Barrier Option Price", f"{barrier_price:.4f}")

    # Plots
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Paths with barrier
    for i in range(min(20, paths)):
        ax1.plot(time, S_paths[i, :], lw=0.8, alpha=0.5)
    ax1.plot(time, np.mean(S_paths, axis=0), 'k--', lw=2, label="Mean Path")
    ax1.axhline(y=Su, color='red', linestyle='--', label=f'Barrier Level Su = {Su}')
    ax1.set_title(" Simulated Asset  Paths with Barrier")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Asset Price")

    # Payoff diagram
    S_range = np.linspace(S0 * 0.5, S0 * 1.5, 100)
    payoff = np.where(S_range > K_digital, 1, 0)
    ax2.plot(S_range, payoff, label='Digital Payoff')
    ax2.axvline(Su, color='r', linestyle='--', label='Barrier Level')
    ax2.hist(S_T, bins=50, density=True, alpha=0.3, color='green', label="Histogram of S(T)")
    ax2.set_title("Payoff Diagram")
    ax2.set_xlabel("Terminal Price")
    ax2.set_ylabel("Payoff")

    st.pyplot(fig2)

    # Additional histograms
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

    ax3.hist(S_paths[:, -1], bins=30, color='skyblue', edgecolor='black')
    ax3.set_title("Terminal Prices Distribution")
    ax3.set_xlabel("Price at Maturity")

    hits = np.any(S_paths > Su, axis=1)
    ax4.bar(['Knocked Out', 'Active'], [np.sum(hits), len(hits) - np.sum(hits)],
            color=['red', 'green'])
    ax4.set_title("Barrier Activation Status")

    st.pyplot(fig3)