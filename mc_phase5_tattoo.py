import subprocess
import numpy as np

np.random.seed(123)

N_CHIPS = 100
K_PAIRS = 4              # validated redundant array size from Phase 4
VTH0_REF_NOMINAL = 0.46893
DVTH_TATTOO_MV = 60.0    # intentional asymmetric-stress target (strong, clear bias)

W_UM = 3.2
L_UM = 0.045
AVT_MV_UM = 4.5
SIGMA_VTH_V = (AVT_MV_UM / np.sqrt(W_UM * L_UM)) / 1000.0
print(f"Design point: W={W_UM}um, sigma_Vth={SIGMA_VTH_V*1000:.2f}mV, K={K_PAIRS} pairs/chip, "
      f"N_CHIPS={N_CHIPS}, tattoo target dvth={DVTH_TATTOO_MV}mV")

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

# Each chip gets ONE true physical tattoo realization: K independent mismatch draws
# (Pelgrom-distributed) around the SAME intended stress target. This is the real,
# deterministic, simulated part -- differences between chips here are genuine
# manufacturing-mismatch physics, not assumed.
all_i_ref = np.zeros((N_CHIPS, K_PAIRS))
all_i_target = np.zeros((N_CHIPS, K_PAIRS))

dvth_tattoo_v = DVTH_TATTOO_MV / 1000.0

for chip in range(N_CHIPS):
    for pair in range(K_PAIRS):
        vth0_ref = VTH0_REF_NOMINAL + np.random.normal(0, SIGMA_VTH_V)
        vth0_target = VTH0_REF_NOMINAL + dvth_tattoo_v + np.random.normal(0, SIGMA_VTH_V)
        i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM)
        all_i_ref[chip, pair] = i_ref
        all_i_target[chip, pair] = i_target
    if (chip + 1) % 20 == 0:
        print(f"  {chip+1}/{N_CHIPS} chips done")

np.savez(f"phase5_tattoo_uniqueness_K{K_PAIRS}_w{int(W_UM*1000)}.npz",
          i_ref=all_i_ref, i_target=all_i_target,
          w_um=W_UM, sigma_vth_mv=SIGMA_VTH_V*1000.0, k_pairs=K_PAIRS,
          dvth_tattoo_mv=DVTH_TATTOO_MV)

print(f"\nSaved raw physical simulation to phase5_tattoo_uniqueness_K{K_PAIRS}_w{int(W_UM*1000)}.npz")
print("Run analyze_phase5_uniqueness.py next -- it adds the measurement-noise model")
print("in Python (no new ngspice calls needed) to compute uniqueness vs. reliability.")