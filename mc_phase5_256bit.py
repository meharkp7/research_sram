import subprocess
import numpy as np

np.random.seed(2024)

N_BITS = 256
K_PAIRS = 4
VTH0_REF_NOMINAL = 0.46893
DVTH_BIT1_MV = 60.0   # stressed -> bit=1
DVTH_BIT0_MV = 0.0    # unstressed -> bit=0

W_UM = 3.2
L_UM = 0.045
AVT_MV_UM = 4.5
SIGMA_VTH_V = (AVT_MV_UM / np.sqrt(W_UM * L_UM)) / 1000.0
print(f"Design point: W={W_UM}um, sigma_Vth={SIGMA_VTH_V*1000:.2f}mV, K={K_PAIRS} pairs/bit, N_BITS={N_BITS}")

NETLIST_TEMPLATE = """.include ./45nm_HP_nmos_ref_param.pm
.include ./45nm_HP_nmos_target_param.pm

.param vth0_ref = {vth0_ref:.6f}
.param vth0_target = {vth0_target:.6f}

Mref    d1 0 0 0 nmos_ref    l=45n w={w_nm:.0f}n
Mtarget d2 0 0 0 nmos_target l=45n w={w_nm:.0f}n

Vref    d1 0 1.0
Vtarget d2 0 1.0

.control
op
print i(Vref) i(Vtarget)
.endc
.end
"""

def run_trial(vth0_ref, vth0_target, w_um):
    w_nm = w_um * 1000.0
    netlist = NETLIST_TEMPLATE.format(vth0_ref=vth0_ref, vth0_target=vth0_target, w_nm=w_nm)
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

# The secret message: a fixed random 256-bit key (ground truth)
secret_bits = np.random.randint(0, 2, size=N_BITS)
print(f"Secret key (first 32 bits shown): {''.join(map(str, secret_bits[:32]))}...")

measured_ratio = np.zeros(N_BITS)

for bit_idx in range(N_BITS):
    dvth_mv = DVTH_BIT1_MV if secret_bits[bit_idx] == 1 else DVTH_BIT0_MV
    dvth_v = dvth_mv / 1000.0

    pair_ratios = []
    for pair in range(K_PAIRS):
        vth0_ref = VTH0_REF_NOMINAL + np.random.normal(0, SIGMA_VTH_V)
        vth0_target = VTH0_REF_NOMINAL + dvth_v + np.random.normal(0, SIGMA_VTH_V)
        i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM)
        pair_ratios.append(i_target / i_ref)

    measured_ratio[bit_idx] = np.mean(pair_ratios)

    if (bit_idx + 1) % 32 == 0:
        print(f"  {bit_idx+1}/{N_BITS} bit-cells done")

np.savez("phase5_256bit_readback.npz",
         secret_bits=secret_bits, measured_ratio=measured_ratio,
         w_um=W_UM, sigma_vth_mv=SIGMA_VTH_V*1000.0, k_pairs=K_PAIRS,
         dvth_bit1_mv=DVTH_BIT1_MV, dvth_bit0_mv=DVTH_BIT0_MV)

# --- Decode using midpoint of the two populations actually measured ---
mean1 = measured_ratio[secret_bits == 1].mean()
mean0 = measured_ratio[secret_bits == 0].mean()
threshold = (mean1 + mean0) / 2
print(f"\nBit=1 population mean ratio: {mean1:.4f}")
print(f"Bit=0 population mean ratio: {mean0:.4f}")
print(f"Decision threshold: {threshold:.4f}")

if mean1 > mean0:
    read_bits = (measured_ratio >= threshold).astype(int)
else:
    read_bits = (measured_ratio <= threshold).astype(int)

n_errors = np.sum(read_bits != secret_bits)
ber = n_errors / N_BITS
print(f"\nBit errors (raw process-mismatch only, no measurement noise): {n_errors}/{N_BITS}")
print(f"Raw BER = {ber*100:.3f}%")

print("\nSaved to phase5_256bit_readback.npz")