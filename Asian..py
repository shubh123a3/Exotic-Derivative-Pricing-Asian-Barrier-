import numpy as np
import enum
import matplotlib.pyplot as plt


class OptionType(enum.Enum):
    call = 1.0
    put = -1.0


def payoffvaluation(S, T, r, payoffcall, payoffput, cp):
    if cp == OptionType.call:
        return np.exp(-r * T) * np.mean(payoffcall(S))
    elif cp == OptionType.put:
        return np.exp(-r * T) * np.mean(payoffput(S))
    else:
        raise ValueError("Invalid OptionType provided")


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


def mainCalculation():
    NoOfPaths = 5000
    NoOfSteps = 250
    cp = OptionType.call
    S0 = 100.0
    r = 0.05
    T = 5
    sigma = 0.2

    paths = GeneratePathsGBMEuler(NoOfPaths, NoOfSteps, T, r, sigma, S0)
    S_paths = paths["S"]
    time = paths["time"]
    S_T = S_paths[:, -1]
    K = 100
    payoffcall = lambda S: np.maximum(S - K, 0.0)
    payoffput = lambda S: np.maximum(K - S, 0.0)
    vt0 = payoffvaluation(S_T, T, r, payoffcall, payoffput, cp)
    print("Value of the contract at t0 ={0}".format(vt0))
    A_t = np.mean(S_paths, axis=1)
    value_asian = payoffvaluation(A_t, T, r, payoffcall, payoffput, cp)
    print("Value of the Asian option at t0 ={0}".format(value_asian))
    print('variance of S(T) = {0}'.format(np.var(S_T)))
    print('variance of A(T) = {0}'.format(np.var(A_t)))
    # ---------------- Plotting ----------------

    # Create a figure with a gridspec layout
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)

    # Plot 1 (Top row, full width): Sample asset paths
    ax1 = fig.add_subplot(gs[0, :])
    # Plot a few sample paths (e.g., first 10)
    for i in range(min(10, NoOfPaths)):
        ax1.plot(time, S_paths[i, :], lw=0.8, alpha=0.7)
    # Also plot the mean asset path (average over all simulations at each time)
    ax1.plot(time, np.mean(S_paths, axis=0), 'k--', lw=2, label='Mean Path')
    ax1.set_title("Simulated Asset Paths (European Option)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Asset Price")
    ax1.legend()

    # Plot 2 (Bottom left): Histogram of terminal prices S(T)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(S_T, bins=30, color='skyblue', edgecolor='black')
    ax2.set_title("Histogram of Terminal Prices S(T)")
    ax2.set_xlabel("S(T)")
    ax2.set_ylabel("Frequency")

    # Plot 3 (Bottom right): Histogram of Asian average prices
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(A_t, bins=30, color='salmon', edgecolor='black')
    ax3.set_title("Histogram of Asian Average Prices")
    ax3.set_xlabel("Average Price")
    ax3.set_ylabel("Frequency")

    # Add an overall title with the option prices
    fig.suptitle(f"European Option Price: {vt0:.4f} | Asian Option Price: {value_asian:.4f}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    mainCalculation()

