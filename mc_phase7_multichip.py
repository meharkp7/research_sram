import subprocess
import numpy as np

N_NEW_CHIPS = 15
BITS_PER_CHIP = 64
K_PAIRS = 4
VTH0_REF_NOMINAL = 0.46893
DVTH_BIT1_MV = 60.0
DVTH_BIT0_MV = 0.0
W_UM = 3.2
L_UM = 0.045
AVT_MV_UM = 4.5
SIGMA_VTH_V = (AVT_MV_UM / np.sqrt(W_UM * L_UM)) / 1000.0

print(f"Generating {N_NEW_CHIPS} new independent chips, {BITS_PER_CHIP} bits each, "
      f"K={K_PAIRS} pairs/bit, W={W_UM}um, sigma_Vth={SIGMA_VTH_V*1000:.2f}mV")
print(f"Total ngspice calls: {N_NEW_CHIPS * BITS_PER_CHIP * K_PAIRS}\n")

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

all_chip_idx, all_bit_pos, all_bit_val, all_ratio = [], [], [], []

for chip_num in range(N_NEW_CHIPS):
    secret_bits = np.random.randint(0, 2, size=BITS_PER_CHIP)
    print(f"=== Chip {chip_num+1}/{N_NEW_CHIPS} ===")
    for bit_idx in range(BITS_PER_CHIP):
        dvth_v = (DVTH_BIT1_MV if secret_bits[bit_idx] == 1 else DVTH_BIT0_MV) / 1000.0
        pair_ratios = []
        for pair in range(K_PAIRS):
            vth0_ref = VTH0_REF_NOMINAL + np.random.normal(0, SIGMA_VTH_V)
            vth0_target = VTH0_REF_NOMINAL + dvth_v + np.random.normal(0, SIGMA_VTH_V)
            i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM)
            pair_ratios.append(i_target / i_ref)
        ratio = np.mean(pair_ratios)

        all_chip_idx.append(chip_num)
        all_bit_pos.append(bit_idx)
        all_bit_val.append(secret_bits[bit_idx])
        all_ratio.append(ratio)

    print(f"  {BITS_PER_CHIP}/{BITS_PER_CHIP} bit-cells done")

np.savez("phase7_multichip_tattoo.npz",
         chip_idx=np.array(all_chip_idx), bit_position=np.array(all_bit_pos),
         bit_value=np.array(all_bit_val), ratio=np.array(all_ratio),
         n_chips=N_NEW_CHIPS, bits_per_chip=BITS_PER_CHIP,
         w_um=W_UM, k_pairs=K_PAIRS, sigma_vth_mv=SIGMA_VTH_V*1000.0)

print(f"\nSaved {len(all_ratio)} total samples across {N_NEW_CHIPS} independent chips "
      f"to phase7_multichip_tattoo.npz")