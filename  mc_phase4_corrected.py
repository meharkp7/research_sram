import subprocess
import numpy as np

np.random.seed(42)

N_PER_CATEGORY = 100
VTH0_REF_NOMINAL = 0.46893   # V

# --- Corrected design point, from the W-sweep ---
W_UM = 3.2 
L_UM = 0.045
AVT_MV_UM = 4.5  # Mezzomo et al., 45nm NMOS baseline Pelgrom coefficient
SIGMA_VTH_V = (AVT_MV_UM / np.sqrt(W_UM * L_UM)) / 1000.0
print(f"Design point: W={W_UM}um, L={L_UM}um -> Pelgrom sigma(Vth) = {SIGMA_VTH_V*1000:.2f} mV")

# Phase 4.1 category definitions (dvth range in mV, aging-induced Vth shift)
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

all_i_ref, all_i_target, all_labels, all_dvth = [], [], [], []

for cat_name, (lo_mv, hi_mv) in CATEGORIES.items():
    print(f"\n=== Category: {cat_name} (dvth range {lo_mv}-{hi_mv} mV) ===")
    for trial in range(N_PER_CATEGORY):
        base_dvth_v = np.random.uniform(lo_mv, hi_mv) / 1000.0
        # Both ref and target get independent Pelgrom mismatch draws.
        # Target additionally carries the category's aging-induced shift.
        vth0_ref = VTH0_REF_NOMINAL + np.random.normal(0, SIGMA_VTH_V)
        vth0_target = VTH0_REF_NOMINAL + base_dvth_v + np.random.normal(0, SIGMA_VTH_V)
        i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM)

        all_i_ref.append(i_ref)
        all_i_target.append(i_target)
        all_labels.append(cat_name)
        all_dvth.append(base_dvth_v * 1000.0)

        if (trial + 1) % 25 == 0:
            print(f"  {trial+1}/{N_PER_CATEGORY} done")

all_i_ref = np.array(all_i_ref)
all_i_target = np.array(all_i_target)
all_labels = np.array(all_labels)
all_dvth = np.array(all_dvth)
ratio = all_i_target / all_i_ref

np.savez("phase4_corrected_dataset_w3200.npz",
         i_ref=all_i_ref, i_target=all_i_target,
         label=all_labels, dvth_mv=all_dvth, ratio=ratio,
         w_um=W_UM, sigma_vth_mv=SIGMA_VTH_V*1000.0)

print("\nSaved to phase4_corrected_dataset_w3200.npz")
print(f"Total samples: {len(all_labels)}")
for cat in CATEGORIES:
    mask = all_labels == cat
    print(f"  {cat:10s}: n={mask.sum():3d}, ratio mean={ratio[mask].mean():.4f}, std={ratio[mask].std():.4f}")