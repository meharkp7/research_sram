.include ./45nm_HP_nmos_ref_param.pm
.include ./45nm_HP_nmos_target_param.pm

.param vth0_ref = 0.47521
.param vth0_target = 0.53442

Mref    d1 0 0 0 nmos_ref    l=45n w=200n
Mtarget d2 0 0 0 nmos_target l=45n w=200n

Vref    d1 0 1.0
Vtarget d2 0 1.0

.control
op
print i(Vref) i(Vtarget)
.endc
.end
