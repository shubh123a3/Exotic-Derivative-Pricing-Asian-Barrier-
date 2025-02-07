import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import enum


class OptionType(enum.Enum):
    call = 1.0
    put = -1.0


def digtalpayoffValuation(S, T, r, payoffcall, payoffput, cp):
    if cp == OptionType.call:
        return np.exp(-r * T) * np.mean(payoffcall(S))
    elif cp == OptionType.put:
        return np.exp(-r * T) * np.mean(payoffput(S))
    else:
        raise ValueError("Invalid OptionType provided")
    # monte carlo


def GeneratePathsGBMEuler(NoOfPaths, NoOfSteps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])

    # Euler Approximation
    S1 = np.zeros([NoOfPaths, NoOfSteps + 1])
    S1[:, 0] = S_0

    time = np.zeros([NoOfSteps + 1])

    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]

        S1[:, i + 1] = S1[:, i] + r * S1[:, i] * dt + sigma * S1[:, i] * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt

    # Retun S1 and S2
    paths = {"time": time, "S": S1}
    return paths


def UpandOutBarrier(S, T, r, payoffcall, payoffput, cp, su):
    # handling barrer
    n1, n2 = S.shape
    barrier = np.zeros([n1, n2]) + su
    if cp == OptionType.call:
        hitM = S > barrier
        hitVec = np.sum(hitM, axis=1)
        hitVec = (hitVec == 0.0).astype(int)
        V_0 = np.exp(-r * T) * np.mean(payoffcall(S[:, -1] * hitVec))
    elif cp == OptionType.call:
        hitM = S < barrier
        hitVec = np.sum(hitM, axis=1)
        hitVec = (hitVec == 0.0).astype(int)
        V_0 = np.exp(-r * T) * np.mean(payoffcall(S[:, -1] * hitVec))
    else:
        raise ValueError("Invalid OptionType provided")
    return V_0


def mainCalculation():
    NoOfPaths = 10000
    NoOfSteps = 250
    cp = OptionType.call
    S0 = 100.0
    r = 0.05
    T = 5
    sigma = 0.2
    Su = 150
    paths = GeneratePathsGBMEuler(NoOfPaths, NoOfSteps, T, r, sigma, S0)
    S_paths = paths["S"]
    S_T = S_paths[:, -1]
    time = paths["time"]
    # payoff setting
    K = 100

    payoffcall = lambda S: np.maximum(S - K, 0.0)  # - np.maximum(S-K2,0)
    payoffput = lambda S: np.maximum(K - S, 0.0)
    S_T_grid = np.linspace(50, S0 * 1.5, 200)
    if cp == OptionType.call:
        plt.plot(S_T_grid, payoffcall(S_T_grid))
    elif cp == OptionType.put:
        plt.plot(S_T_grid, payoffput(S_T_grid))
    else:
        raise ValueError("Invalid OptionType provided")
    val_to = digtalpayoffValuation(S_T, T, r, payoffcall, payoffput, cp)
    print("Value of the contract at t0 ={0}".format(val_to))
    barrier_price = UpandOutBarrier(S_paths, T, r, payoffcall, payoffput, cp, Su)
    print("Value of the barrier contract at t0 ={0}".format(barrier_price))

    # Mark the barrier level on the payoff plot as a vertical line
    plt.axvline(x=Su, color='red', linestyle='--', label=f'Barrier Level Su = {Su}')
    # Overlay a histogram of terminal asset prices
    plt.hist(S_T, bins=50, density=True, alpha=0.3, color='gray', label="Histogram of S(T)")
    plt.title("Digital Option Payoff with Barrier Level")
    plt.xlabel("Underlying Asset Price")
    plt.ylabel("Payoff / Density")
    plt.legend()
    plt.grid(True)

    # Calculate digital option price (without barrier) and barrier option price
    digital_price = digtalpayoffValuation(S_T, T, r, payoffcall, payoffput, cp)
    barrier_price = UpandOutBarrier(S_paths, T, r, payoffcall, payoffput, cp, Su)
    print("Value of the digital contract at t0 = {:.4f}".format(digital_price))
    print("Value of the barrier contract at t0 = {:.4f}".format(barrier_price))

    # Plot simulated asset paths with the barrier level
    plt.figure(figsize=(12, 8))
    for i in range(min(20, NoOfPaths)):  # Plot 20 sample paths for clarity
        plt.plot(time, S_paths[i, :], lw=0.8, alpha=0.5)
    plt.plot(time, np.mean(S_paths, axis=0), 'k--', lw=2, label="Mean Path")
    plt.axhline(y=Su, color='red', linestyle='--', label=f'Barrier Level Su = {Su}')
    plt.title("Simulated Asset Paths with Barrier Level")
    plt.xlabel("Time")
    plt.ylabel("Asset Price")
    plt.legend()
    plt.grid(True)

    # Bar plot comparing the digital option price and barrier option price
    plt.figure(figsize=(8, 6))
    labels = ['Digital Option', 'Barrier Option']
    prices = [digital_price, barrier_price]
    bar_colors = ['blue', 'orange']
    bars = plt.bar(labels, prices, color=bar_colors)
    plt.title("Option Price Comparison: Digital vs Barrier Option")
    plt.ylabel("Option Price")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()


if __name__ == "__main__":
    mainCalculation()

mainCalculation()


