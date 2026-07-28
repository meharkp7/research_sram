import subprocess
import numpy as np

np.random.seed(42)

N_TRIALS = 100
VTH0_REF_NOMINAL = 0.46893
VTH0_TARGET_NOMINAL = 0.52893   # +60mV NBTI shift ("severely aged")

# Real geometry -- must match what's in the netlist below
W_UM = 0.8   # 200nm
L_UM = 0.045 # 45nm

# Pelgrom coefficient for 45nm NMOS, baseline/reference process
# Mezzomo et al., "Pockets Engineering Impact on Mismatch Performance
# on 45nm MOSFET Technologies" -- AVT ~= 4.5 mV.um
AVT_MV_UM = 4.5
SIGMA_VTH_V = (AVT_MV_UM / np.sqrt(W_UM * L_UM)) / 1000.0  # convert mV -> V

print(f"Pelgrom-predicted sigma(Vth) at W={W_UM}um, L={L_UM}um: {SIGMA_VTH_V*1000:.2f} mV")

NETLIST_TEMPLATE = """.include ./45nm_HP_nmos_ref_param.pm
.include ./45nm_HP_nmos_target_param.pm

.param vth0_ref = {vth0_ref:.6f}
.param vth0_target = {vth0_target:.6f}

Mref    d1 0 0 0 nmos_ref    l=45n w=800n
Mtarget d2 0 0 0 nmos_target l=45n w=800n

Vref    d1 0 1.0
Vtarget d2 0 1.0

.control
op
print i(Vref) i(Vtarget)
.endc
.end
"""

def run_trial(vth0_ref, vth0_target):
    netlist = NETLIST_TEMPLATE.format(vth0_ref=vth0_ref, vth0_target=vth0_target)
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

i_refs, i_targets = [], []
for trial in range(N_TRIALS):
    vth0_ref = VTH0_REF_NOMINAL + np.random.normal(0, SIGMA_VTH_V)
    vth0_target = VTH0_TARGET_NOMINAL + np.random.normal(0, SIGMA_VTH_V)
    i_ref, i_target = run_trial(vth0_ref, vth0_target)
    i_refs.append(i_ref)
    i_targets.append(i_target)
    if (trial + 1) % 20 == 0:
        print(f"  completed {trial+1}/{N_TRIALS} trials")

i_refs = np.array(i_refs)
i_targets = np.array(i_targets)
diff = i_refs - i_targets
ratio = i_targets / i_refs

def cv(x):
    return 100 * np.std(x) / np.mean(x)

print(f"\n=== Results across {N_TRIALS} trials (Pelgrom-scaled mismatch, min size) ===")
print(f"I_ref    : mean={np.mean(i_refs)*1e9:.4f} nA, std={np.std(i_refs)*1e9:.4f} nA, CV={cv(i_refs):.2f}%")
print(f"I_target : mean={np.mean(i_targets)*1e9:.4f} nA, std={np.std(i_targets)*1e9:.4f} nA, CV={cv(i_targets):.2f}%")
print(f"Raw diff : mean={np.mean(diff)*1e9:.4f} nA, std={np.std(diff)*1e9:.4f} nA, CV={cv(diff):.2f}%")
print(f"Ratio    : mean={np.mean(ratio):.4f}, std={np.std(ratio):.4f}, CV={cv(ratio):.2f}%")

np.savez("mc_mismatch_pelgrom_800nm.npz", i_refs=i_refs, i_targets=i_targets, diff=diff, ratio=ratio)
print("\nSaved to mc_mismatch_pelgrom_800nm.npz")