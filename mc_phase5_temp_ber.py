import subprocess
import numpy as np

np.random.seed(2024)  # same seed as Segment 5.2 -- reproduces the identical 256-bit secret key

N_BITS = 256
K_PAIRS = 4
VTH0_REF_NOMINAL = 0.46893
DVTH_BIT1_MV = 60.0
DVTH_BIT0_MV = 0.0
W_UM = 3.2
L_UM = 0.045
AVT_MV_UM = 4.5
SIGMA_VTH_V = (AVT_MV_UM / np.sqrt(W_UM * L_UM)) / 1000.0

CALIBRATION_TEMP_C = 25
FIELD_TEMP_C = 75  # worst-case realistic operating swing

NETLIST_TEMPLATE = """.include ./45nm_HP_nmos_ref_param.pm
.include ./45nm_HP_nmos_target_param.pm

.param vth0_ref = {vth0_ref:.6f}
.param vth0_target = {vth0_target:.6f}

Mref    d1 0 0 0 nmos_ref    l=45n w={w_nm:.0f}n
Mtarget d2 0 0 0 nmos_target l=45n w={w_nm:.0f}n

Vref    d1 0 1.0
Vtarget d2 0 1.0

.temp {temp_c}

.control
op
print i(Vref) i(Vtarget)
.endc
.end
"""

def run_trial(vth0_ref, vth0_target, w_um, temp_c):
    w_nm = w_um * 1000.0
    netlist = NETLIST_TEMPLATE.format(vth0_ref=vth0_ref, vth0_target=vth0_target,
                                       w_nm=w_nm, temp_c=temp_c)
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

def measure_key(secret_bits, temp_c, label):
    print(f"\n=== Measuring at {temp_c}C ({label}) ===")
    measured_ratio = np.zeros(N_BITS)
    for bit_idx in range(N_BITS):
        dvth_mv = DVTH_BIT1_MV if secret_bits[bit_idx] == 1 else DVTH_BIT0_MV
        dvth_v = dvth_mv / 1000.0
        pair_ratios = []
        for pair in range(K_PAIRS):
            vth0_ref = VTH0_REF_NOMINAL + np.random.normal(0, SIGMA_VTH_V)
            vth0_target = VTH0_REF_NOMINAL + dvth_v + np.random.normal(0, SIGMA_VTH_V)
            i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM, temp_c)
            pair_ratios.append(i_target / i_ref)
        measured_ratio[bit_idx] = np.mean(pair_ratios)
        if (bit_idx + 1) % 64 == 0:
            print(f"  {bit_idx+1}/{N_BITS} bit-cells done")
    return measured_ratio

# Same secret key as Segment 5.2 (same seed, same draw order)
secret_bits = np.random.randint(0, 2, size=N_BITS)
print(f"Secret key (first 32 bits): {''.join(map(str, secret_bits[:32]))}...")

# --- Calibration pass: establishes the threshold ---
ratio_calib = measure_key(secret_bits, CALIBRATION_TEMP_C, "calibration")
mean1_calib = ratio_calib[secret_bits == 1].mean()
mean0_calib = ratio_calib[secret_bits == 0].mean()
threshold = (mean1_calib + mean0_calib) / 2
higher_is_1 = mean1_calib > mean0_calib
print(f"\nCalibration threshold (fixed at {CALIBRATION_TEMP_C}C): {threshold:.4f}")

read_bits_calib = (ratio_calib >= threshold).astype(int) if higher_is_1 else (ratio_calib <= threshold).astype(int)
ber_calib = np.mean(read_bits_calib != secret_bits)
print(f"BER at calibration temp itself (process mismatch only): {ber_calib*100:.3f}%")

# --- Field pass: same key, different temperature, SAME fixed threshold ---
ratio_field = measure_key(secret_bits, FIELD_TEMP_C, "field read")
read_bits_field = (ratio_field >= threshold).astype(int) if higher_is_1 else (ratio_field <= threshold).astype(int)
ber_field = np.mean(read_bits_field != secret_bits)
print(f"\nBER at field temp ({FIELD_TEMP_C}C) using threshold fixed at {CALIBRATION_TEMP_C}C: {ber_field*100:.3f}%")

np.savez("phase5_temp_ber_test.npz",
          secret_bits=secret_bits, ratio_calib=ratio_calib, ratio_field=ratio_field,
          threshold=threshold, ber_calib=ber_calib, ber_field=ber_field,
          calib_temp=CALIBRATION_TEMP_C, field_temp=FIELD_TEMP_C)
print("\nSaved to phase5_temp_ber_test.npz")