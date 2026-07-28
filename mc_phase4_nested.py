import subprocess
import numpy as np

np.random.seed(42)

N_CHIPS_PER_CATEGORY = 40
K_PAIRS = 4          # redundant sensor pairs per chip, sharing the same true aging state
VTH0_REF_NOMINAL = 0.46893

W_UM = 3.2
L_UM = 0.045
AVT_MV_UM = 4.5
SIGMA_VTH_V = (AVT_MV_UM / np.sqrt(W_UM * L_UM)) / 1000.0
print(f"Design point: W={W_UM}um, sigma_Vth={SIGMA_VTH_V*1000:.2f}mV, K={K_PAIRS} pairs/chip, "
      f"{N_CHIPS_PER_CATEGORY} chips/category")

CATEGORIES = {
    "fresh":    (0.0, 0.0),
    "light":    (5.0, 20.0),
    "moderate": (20.0, 45.0),
    "severe":   (45.0, 70.0),
}

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

all_chip_ratio, all_label, all_dvth = [], [], []

for cat_name, (lo_mv, hi_mv) in CATEGORIES.items():
    print(f"\n=== Category: {cat_name} (dvth range {lo_mv}-{hi_mv} mV) ===")
    for chip in range(N_CHIPS_PER_CATEGORY):
        # ONE true aging state for this physical chip
        dvth_chip_v = np.random.uniform(lo_mv, hi_mv) / 1000.0

        pair_ratios = []
        for pair in range(K_PAIRS):
            # K independent mismatch draws sharing the same dvth_chip
            vth0_ref = VTH0_REF_NOMINAL + np.random.normal(0, SIGMA_VTH_V)
            vth0_target = VTH0_REF_NOMINAL + dvth_chip_v + np.random.normal(0, SIGMA_VTH_V)
            i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM)
            pair_ratios.append(i_target / i_ref)

        chip_ratio = np.mean(pair_ratios)
        all_chip_ratio.append(chip_ratio)
        all_label.append(cat_name)
        all_dvth.append(dvth_chip_v * 1000.0)

        if (chip + 1) % 10 == 0:
            print(f"  {chip+1}/{N_CHIPS_PER_CATEGORY} chips done")

all_chip_ratio = np.array(all_chip_ratio)
all_label = np.array(all_label)
all_dvth = np.array(all_dvth)

np.savez(f"phase4_nested_K{K_PAIRS}_w{int(W_UM*1000)}.npz",
         chip_ratio=all_chip_ratio, label=all_label, dvth_mv=all_dvth,
         w_um=W_UM, sigma_vth_mv=SIGMA_VTH_V*1000.0, k_pairs=K_PAIRS)

print(f"\nSaved to phase4_nested_K{K_PAIRS}_w{int(W_UM*1000)}.npz")
for cat in CATEGORIES:
    mask = all_label == cat
    print(f"  {cat:10s}: n={mask.sum():3d}, chip_ratio mean={all_chip_ratio[mask].mean():.4f}, "
          f"std={all_chip_ratio[mask].std():.4f}")