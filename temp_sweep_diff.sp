* Differential leakage: fresh reference vs NBTI-aged target, real BSIM4 physics
.include ./spice_model_collections/ptm/45nm_HP.pm
.include ./45nm_HP_nmos_aged.pm

Mref    d1 0 0 0 nmos       l=45n w=200n
Mtarget d2 0 0 0 nmos_aged  l=45n w=200n

Vref    d1 0 1.0
Vtarget d2 0 1.0

.control
dc TEMP 0 100 10
print v(d1) i(Vref) v(d2) i(Vtarget)
.endc
.end
