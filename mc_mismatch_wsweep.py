import subprocess
import numpy as np

np.random.seed(42)

N_TRIALS = 100
VTH0_REF_NOMINAL = 0.46893
VTH0_TARGET_NOMINAL = 0.52893   # +60mV NBTI shift ("severely aged")

L_UM = 0.045  # 45nm, fixed

# Pelgrom coefficient for 45nm NMOS, baseline/reference process
# Mezzomo et al., "Pockets Engineering Impact on Mismatch Performance
# on 45nm MOSFET Technologies" -- AVT ~= 4.5 mV.um
AVT_MV_UM = 4.5

# Sweep W: minimum size up through progressively larger devices
W_SWEEP_UM = [0.2, 0.4, 0.8, 1.6, 3.2, 6.4]

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

def cv(x):
    return 100 * np.std(x) / np.mean(x)

def d_prime(a, b):
    return abs(np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)

summary_rows = []

for w_um in W_SWEEP_UM:
    sigma_vth_v = (AVT_MV_UM / np.sqrt(w_um * L_UM)) / 1000.0
    print(f"\n=== W={w_um}um, L={L_UM}um -> Pelgrom sigma(Vth) = {sigma_vth_v*1000:.2f} mV ===")

    i_refs, i_targets = [], []
    for trial in range(N_TRIALS):
        vth0_ref = VTH0_REF_NOMINAL + np.random.normal(0, sigma_vth_v)
        vth0_target = VTH0_TARGET_NOMINAL + np.random.normal(0, sigma_vth_v)
        i_ref, i_target = run_trial(vth0_ref, vth0_target, w_um)
        i_refs.append(i_ref)
        i_targets.append(i_target)

    i_refs = np.array(i_refs)
    i_targets = np.array(i_targets)
    dprime = d_prime(i_refs, i_targets)

    print(f"  I_ref    : mean={np.mean(i_refs)*1e9:.4f} nA, CV={cv(i_refs):.2f}%")
    print(f"  I_target : mean={np.mean(i_targets)*1e9:.4f} nA, CV={cv(i_targets):.2f}%")
    print(f"  d' (fresh vs severe, single pair) = {dprime:.3f}")

    fname = f"mc_mismatch_pelgrom_{int(w_um*1000)}nm.npz"
    np.savez(fname, i_refs=i_refs, i_targets=i_targets, w_um=w_um, sigma_vth_v=sigma_vth_v)
    summary_rows.append((w_um, sigma_vth_v*1000, dprime, np.mean(i_refs)*1e9, np.mean(i_targets)*1e9))
    print(f"  Saved to {fname}")

print("\n=== SUMMARY ===")
print(f"{'W (um)':>8} {'sigma_Vth (mV)':>16} {'d prime':>10}")
for w_um, sig, dp, mr, mt in summary_rows:
    print(f"{w_um:8.2f} {sig:16.2f} {dp:10.3f}")

np.savez("wsweep_summary.npz",
         w_list=[r[0] for r in summary_rows],
         sigma_list=[r[1] for r in summary_rows],
         dprime_list=[r[2] for r in summary_rows])
print("\nSaved summary to wsweep_summary.npz")