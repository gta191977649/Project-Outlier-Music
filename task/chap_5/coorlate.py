import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Generate random signals
rng = np.random.default_rng()
x = rng.standard_normal(1000)
y = np.concatenate([rng.standard_normal(100), x])

# Compute cross-correlation and lags
correlation = signal.correlate(x, y, mode="full")
lags = signal.correlation_lags(x.size, y.size, mode="full")
lag = lags[np.argmax(correlation)]

# Plotting
plt.figure(figsize=(12, 6))

# Plot signal x
plt.subplot(2, 1, 1)
plt.plot(x, label='Signal x')
plt.title('Signal x')
plt.legend()

# Plot signal y
plt.subplot(2, 1, 2)
plt.plot
