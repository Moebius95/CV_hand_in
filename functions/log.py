import numpy as np
import matplotlib.pyplot as plt
import cmath
"""
MAX_FREQ = 1_000_000


def transfer(R1, R2, C1, C2, w):
    compl = R1 / (R1 + 1 / (w * C1 * 1j)) * 1 / (w * R2 * C2 * 1j + 1)
    return np.sqrt(np.real(compl) ** 2 + np.imag(compl) ** 2)


input = np.linspace(1, MAX_FREQ, 10_000)
fig, ax = plt.subplots()
plt.semilogx([x / (2 * np.pi) for x in input],
             [20 * np.log10(transfer(13, 12, 620 * 10 ** (-6), 820 * 10 ** (-9), x)) for x in input],
             label="TEST")
plt.vlines([20, 16_000], -40, 1, colors='r', linestyles='dashed', label="cut-off frequencies")
plt.hlines(-3, 0, 1_000_000, colors='b', linestyles='dashed')
plt.ylabel("attenuation [dB]")
plt.xlabel("frequency [Hz]")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.plot(20, -3, 'gx', markersize=10)
plt.plot(16_000, -3, 'gx', markersize=10)
ax.text(20, -41, "f_low=20Hz", fontsize=11,
        verticalalignment='top', bbox=props)
ax.text(16000, -41, "f_high=16kHz", fontsize=11,
        verticalalignment='top', bbox=props)
ax.text(0.1, -4, "-3dB attenuation", fontsize=11,
        verticalalignment='top', bbox=props)

plt.grid()
# plt.legend(loc='center right')
plt.show()
"""
print(np.logspace(1, np.log10(40_000), num=10, endpoint=True))
input = [10, 25, 63, 159, 399, 1_002, 2_520, 6_333, 15_916, 40_000, 4_000, 8_000, 10_000, 13_000, 20_000, 30_000]
output = [2.983, 2.983, 2.985, 2.981, 2.980, 2.977, 2.934, 2.769, 2.134, 1.138, 2.887, 2.665, 2.535, 2.338, 1.903, 1.434]
plt.semilogx(input, [20*np.log10(x/3) for x in output], 'rx', clip_on=False)
db_fit = [20*np.log10(x/3) for x in output[8:10]]
print(db_fit)
m, b = np.polyfit(np.log10(input[8:10]), db_fit, deg=1)
plt.plot(range(8000, 50_000), [m * np.log10(x) + b for x in range(8000, 50_000)], color='grey', linestyle='dashed', label='interpolation', clip_on=False)
plt.hlines([-3, 0], 6, 40_000, colors=['b', 'grey'], linestyles='dashed', label='-3dB', clip_on=False)
plt.vlines(16_000, -10, 1, colors=['b'], linestyles='dashed', clip_on=False)
plt.vlines(9660, -10, 1, colors=['grey'], linestyles='dashed', clip_on=False)
print(m, b)
plt.plot(9660, -10, 'rx', markersize=10, label='interpolated cut-off', clip_on=False)
plt.plot(16_000, -10, 'gx', markersize=10, label='actual cut-off', clip_on=False)
plt.legend()
plt.grid(axis='x')
plt.grid(axis='y', which='both')
plt.xlabel("f [Hz]")
plt.ylabel("attenuation [dB]")
plt.show()

