import subprocess
import numpy as np

np.random.seed(777)

N_CHIPS = 20
BITS_PER_CHIP = 64
K_PAIRS = 4
VTH0_REF_NOMINAL = 0.46893
DVTH_BIT1_MV = 60.0
DVTH_BIT0_MV = 0.0
W_UM = 3.2
L_UM = 0.045
AVT_MV_UM = 4.5

# Standard mismatch decomposition: total variance = local (independent, Pelgrom) + systematic
# (die-level, spatially correlated). We use a simple model: each chip gets ONE random systematic
# offset (shared by all its transistors), plus each transistor still gets its own independent
# local draw. This is a standard simplification of within-die vs die-to-die variation.
SIGMA_LOCAL_V = (AVT_MV_UM / np.sqrt(W_UM * L_UM)) / 1000.0  # same as before, ~11.86mV
# Systematic component magnitude: literature-typical split is roughly comparable magnitude to
# local variation for a well-controlled process; we use 50% of the local sigma as a moderate,
# clearly-stated assumption (not derived from a specific citation -- flag this explicitly).
SIGMA_SYSTEMATIC_V = 0.5 * SIGMA_LOCAL_V

print(f"Local (independent) sigma: {SIGMA_LOCAL_V*1000:.2f}mV")
print(f"Systematic (per-chip, correlated) sigma: {SIGMA_SYSTEMATIC_V*1000:.2f}mV "
      f"(ASSUMPTION: 50% of local -- not literature-derived, stated explicitly as a limitation)")

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

for chip_num in range(N_CHIPS):
    # ONE systematic offset per chip, shared by every transistor on that chip
    chip_systematic_offset_v = np.random.normal(0, SIGMA_SYSTEMATIC_V)

    secret_bits = np.random.randint(0, 2, size=BITS_PER_CHIP)
    print(f"=== Chip {chip_num+1}/{N_CHIPS} (systematic offset={chip_systematic_offset_v*1000:.2f}mV) ===")
    for bit_idx in range(BITS_PER_CHIP):
        dvth_v = (DVTH_BIT1_MV if secret_bits[bit_idx] == 1 else DVTH_BIT0_MV) / 1000.0
        pair_ratios = []
        for pair in range(K_PAIRS):
            # Local (independent) + systematic (shared per-chip) components combined
            vth0_ref = (VTH0_REF_NOMINAL + chip_systematic_offset_v
                        + np.random.normal(0, SIGMA_LOCAL_V))
            vth0_target = (VTH0_REF_NOMINAL + dvth_v + chip_systematic_offset_v
                           + np.random.normal(0, SIGMA_LOCAL_V))
            i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM)
            pair_ratios.append(i_target / i_ref)
        ratio = np.mean(pair_ratios)

        all_chip_idx.append(chip_num)
        all_bit_pos.append(bit_idx)
        all_bit_val.append(secret_bits[bit_idx])
        all_ratio.append(ratio)

    print(f"  {BITS_PER_CHIP}/{BITS_PER_CHIP} bit-cells done")

np.savez("phase8_correlated_mismatch.npz",
         chip_idx=np.array(all_chip_idx), bit_position=np.array(all_bit_pos),
         bit_value=np.array(all_bit_val), ratio=np.array(all_ratio),
         n_chips=N_CHIPS, bits_per_chip=BITS_PER_CHIP,
         sigma_local_mv=SIGMA_LOCAL_V*1000, sigma_systematic_mv=SIGMA_SYSTEMATIC_V*1000)

print(f"\nSaved {len(all_ratio)} samples across {N_CHIPS} chips (WITH systematic component) "
      f"to phase8_correlated_mismatch.npz")