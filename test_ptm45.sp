* Sanity check: PTM 45nm HP NMOS Id-Vgs sweep, real BSIM4 subthreshold physics
.include ./spice_model_collections/ptm/45nm_HP.pm

M1 d g 0 0 nmos l=45n w=200n
VD d 0 1.0
VG g 0 0

.control
dc VG 0 1.0 0.02
print v(d) i(VD)
.endc
.end