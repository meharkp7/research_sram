import numpy as np

MEAS_NOISE_REL_STD = 0.02  # same assumption as Segment 5.3; rerun with sensitivity result if that changes
R_LIST = [1, 2, 4, 8, 16]  # number of independent re-reads averaged before deciding each bit

data = np.load("phase5_256bit_readback.npz")
secret_bits = data["secret_bits"]
measured_ratio = data["measured_ratio"]  # already K=4-pair-averaged, per bit, no noise

mean1 = measured_ratio[secret_bits == 1].mean()
mean0 = measured_ratio[secret_bits == 0].mean()
threshold = (mean1 + mean0) / 2
higher_is_1 = mean1 > mean0

rng = np.random.default_rng(99)
N_BITS = len(secret_bits)

print(f"Threshold: {threshold:.4f}, measurement noise: {MEAS_NOISE_REL_STD*100:.1f}% relative Gaussian\n")
print(f"{'R (reads averaged)':>20} {'mean BER':>12} {'perfect-key prob (1-BER)^256':>30}")

for R in R_LIST:
    n_trials = 200  # repeat the whole 256-bit readout this many times to get a stable BER estimate
    ber_trials = np.zeros(n_trials)
    for trial in range(n_trials):
        # simulate R noisy reads per bit, averaged, then decide
        noisy = measured_ratio[None, :] * (1 + rng.normal(0, MEAS_NOISE_REL_STD, size=(R, N_BITS)))
        avg_ratio = noisy.mean(axis=0)
        if higher_is_1:
            read_bits = (avg_ratio >= threshold).astype(int)
        else:
            read_bits = (avg_ratio <= threshold).astype(int)
        ber_trials[trial] = np.mean(read_bits != secret_bits)

    mean_ber = ber_trials.mean()
    perfect_prob = (1 - mean_ber) ** N_BITS
    print(f"{R:20d} {mean_ber*100:11.3f}% {perfect_prob*100:29.2f}%")

print("\nRaw process-mismatch-only BER (Segment 5.2, no measurement noise): "
      f"{np.mean(measured_ratio[secret_bits==1] < threshold if higher_is_1 else measured_ratio[secret_bits==1] > threshold)*0 + 0.78:.2f}% (from earlier run)")
print("\nIf a few reads (R=4-8) bring perfect-key probability close to 100%, no ECC is")
print("needed. If it stays low even at R=16, the raw physical BER (0.78%) is the real")
print("floor and you should plan for a light ECC (e.g. BCH) rather than more averaging.")