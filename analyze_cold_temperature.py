import subprocess
import numpy as np

np.random.seed(2024)  # same secret key as Segment 5.2/5.6 for direct comparability

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
COLD_TEMPS_C = [10, 5, 0]  # sweep toward the cold end that was never isolated before

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
    with open("_cold_trial.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_cold_trial.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    return i_ref, i_target

secret_bits = np.random.randint(0, 2, size=N_BITS)

# Fixed mismatch per bit-cell, drawn ONCE -- same physical chip across all temperatures tested
fixed_vth0_ref = np.zeros((N_BITS, K_PAIRS))
fixed_vth0_target = np.zeros((N_BITS, K_PAIRS))
for bit_idx in range(N_BITS):
    dvth_v = (DVTH_BIT1_MV if secret_bits[bit_idx] == 1 else DVTH_BIT0_MV) / 1000.0
    for pair in range(K_PAIRS):
        fixed_vth0_ref[bit_idx, pair] = VTH0_REF_NOMINAL + np.random.normal(0, SIGMA_VTH_V)
        fixed_vth0_target[bit_idx, pair] = VTH0_REF_NOMINAL + dvth_v + np.random.normal(0, SIGMA_VTH_V)

def measure_at_temp(temp_c):
    measured_ratio = np.zeros(N_BITS)
    for bit_idx in range(N_BITS):
        pair_ratios = []
        for pair in range(K_PAIRS):
            i_ref, i_target = run_trial(fixed_vth0_ref[bit_idx, pair],
                                          fixed_vth0_target[bit_idx, pair], W_UM, temp_c)
            pair_ratios.append(i_target / i_ref)
        measured_ratio[bit_idx] = np.mean(pair_ratios)
        if (bit_idx + 1) % 64 == 0:
            print(f"    {bit_idx+1}/{N_BITS} done")
    return measured_ratio

print(f"=== Calibration at {CALIBRATION_TEMP_C}C ===")
ratio_calib = measure_at_temp(CALIBRATION_TEMP_C)
mean1 = ratio_calib[secret_bits == 1].mean()
mean0 = ratio_calib[secret_bits == 0].mean()
threshold = (mean1 + mean0) / 2
higher_is_1 = mean1 > mean0
ber_calib = np.mean(((ratio_calib >= threshold) if higher_is_1 else (ratio_calib <= threshold)).astype(int) != secret_bits)
print(f"Threshold: {threshold:.4f}, BER at calibration: {ber_calib*100:.3f}%\n")

print(f"=== Cold sweep, SAME fixed chip, threshold fixed at {CALIBRATION_TEMP_C}C ===")
for cold_temp in COLD_TEMPS_C:
    print(f"  Measuring at {cold_temp}C...")
    ratio_cold = measure_at_temp(cold_temp)
    read_bits = (ratio_cold >= threshold).astype(int) if higher_is_1 else (ratio_cold <= threshold).astype(int)
    ber_cold = np.mean(read_bits != secret_bits)
    n_flips = np.sum(read_bits != (((ratio_calib >= threshold) if higher_is_1 else (ratio_calib <= threshold)).astype(int)))
    print(f"  T={cold_temp}C: BER={ber_cold*100:.3f}% ({int(ber_cold*N_BITS)} errors, "
          f"{n_flips} bits flipped vs calibration reading)")

print("\nIf BER climbs as temperature drops toward 0C, this confirms cold-temperature")
print("degradation is real and directional (worse than the hot-side result from Segment 5.6),")
print("not an artifact of this particular compound-test random draw.")