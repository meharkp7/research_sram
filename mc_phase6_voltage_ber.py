import subprocess
import numpy as np

np.random.seed(2024)  # same secret key as Segments 5.2/5.6

N_BITS = 256
K_PAIRS = 4
VTH0_REF_NOMINAL = 0.46893
DVTH_BIT1_MV = 60.0
DVTH_BIT0_MV = 0.0
W_UM = 3.2
L_UM = 0.045
AVT_MV_UM = 4.5
SIGMA_VTH_V = (AVT_MV_UM / np.sqrt(W_UM * L_UM)) / 1000.0

VDD_NOMINAL = 1.0
VDD_LOW = 0.9
VDD_HIGH = 1.1  # both within the validated ~2.4%-residual safe range, well clear of the 1.3x GIDL cliff

NETLIST_TEMPLATE = """.include ./45nm_HP_nmos_ref_param.pm
.include ./45nm_HP_nmos_target_param.pm

.param vth0_ref = {vth0_ref:.6f}
.param vth0_target = {vth0_target:.6f}

Mref    d1 0 0 0 nmos_ref    l=45n w={w_nm:.0f}n
Mtarget d2 0 0 0 nmos_target l=45n w={w_nm:.0f}n

Vref    d1 0 {vdd}
Vtarget d2 0 {vdd}

.control
op
print i(Vref) i(Vtarget)
.endc
.end
"""

def run_trial(vth0_ref, vth0_target, w_um, vdd):
    w_nm = w_um * 1000.0
    netlist = NETLIST_TEMPLATE.format(vth0_ref=vth0_ref, vth0_target=vth0_target,
                                       w_nm=w_nm, vdd=vdd)
    with open("_mc_trial.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_mc_trial.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    if i_ref is None or i_target is None:
        raise RuntimeError(f"Parse failure. Raw output:\n{result.stdout}")
    return i_ref, i_target

secret_bits = np.random.randint(0, 2, size=N_BITS)
print(f"Secret key (first 32 bits): {''.join(map(str, secret_bits[:32]))}...")

# --- Fixed mismatch per bit-cell, drawn ONCE -- this is the physical chip ---
fixed_vth0_ref = np.zeros((N_BITS, K_PAIRS))
fixed_vth0_target = np.zeros((N_BITS, K_PAIRS))
for bit_idx in range(N_BITS):
    dvth_v = (DVTH_BIT1_MV if secret_bits[bit_idx] == 1 else DVTH_BIT0_MV) / 1000.0
    for pair in range(K_PAIRS):
        fixed_vth0_ref[bit_idx, pair] = VTH0_REF_NOMINAL + np.random.normal(0, SIGMA_VTH_V)
        fixed_vth0_target[bit_idx, pair] = VTH0_REF_NOMINAL + dvth_v + np.random.normal(0, SIGMA_VTH_V)

def measure_at_vdd(vdd, label):
    print(f"\n=== Measuring the SAME fixed chip at VDD={vdd}V ({label}) ===")
    measured_ratio = np.zeros(N_BITS)
    for bit_idx in range(N_BITS):
        pair_ratios = []
        for pair in range(K_PAIRS):
            i_ref, i_target = run_trial(fixed_vth0_ref[bit_idx, pair],
                                          fixed_vth0_target[bit_idx, pair], W_UM, vdd)
            pair_ratios.append(i_target / i_ref)
        measured_ratio[bit_idx] = np.mean(pair_ratios)
        if (bit_idx + 1) % 64 == 0:
            print(f"  {bit_idx+1}/{N_BITS} bit-cells done")
    return measured_ratio

ratio_nominal = measure_at_vdd(VDD_NOMINAL, "calibration, nominal VDD")
mean1 = ratio_nominal[secret_bits == 1].mean()
mean0 = ratio_nominal[secret_bits == 0].mean()
threshold = (mean1 + mean0) / 2
higher_is_1 = mean1 > mean0
print(f"\nThreshold (fixed at VDD={VDD_NOMINAL}V): {threshold:.4f}")

def ber_of(ratio_arr):
    read_bits = (ratio_arr >= threshold).astype(int) if higher_is_1 else (ratio_arr <= threshold).astype(int)
    return np.mean(read_bits != secret_bits), read_bits

ber_nominal, bits_nominal = ber_of(ratio_nominal)
print(f"BER at nominal VDD: {ber_nominal*100:.3f}%")

ratio_low = measure_at_vdd(VDD_LOW, "low VDD (undervoltage), SAME chip")
ber_low, bits_low = ber_of(ratio_low)
n_flip_low = np.sum(bits_low != bits_nominal)
print(f"BER at VDD={VDD_LOW}V: {ber_low*100:.3f}% ({n_flip_low} bits flipped vs nominal)")

ratio_high = measure_at_vdd(VDD_HIGH, "high VDD (overvoltage), SAME chip")
ber_high, bits_high = ber_of(ratio_high)
n_flip_high = np.sum(bits_high != bits_nominal)
print(f"BER at VDD={VDD_HIGH}V: {ber_high*100:.3f}% ({n_flip_high} bits flipped vs nominal)")

np.savez("phase6_voltage_ber_test.npz",
         secret_bits=secret_bits, ratio_nominal=ratio_nominal,
         ratio_low=ratio_low, ratio_high=ratio_high, threshold=threshold,
         vdd_nominal=VDD_NOMINAL, vdd_low=VDD_LOW, vdd_high=VDD_HIGH,
         ber_nominal=ber_nominal, ber_low=ber_low, ber_high=ber_high)
print("\nSaved to phase6_voltage_ber_test.npz")