import numpy as np
import pandas as pd
from datetime import date
from schema import LeakageSample
from pydantic import ValidationError

TODAY = date.today().isoformat()
validated_rows = []
errors = []

def add_row(**kwargs):
    try:
        sample = LeakageSample(**kwargs)
        validated_rows.append(sample.model_dump())
    except ValidationError as e:
        errors.append((kwargs.get("sample_id", "UNKNOWN"), str(e)))

# --- Phase 4: validated nested aging-category dataset (W=3.2um, K=4) ---
CATEGORY_MIDPOINT_MV = {"fresh": 0.0, "light": 12.5, "moderate": 32.5, "severe": 57.5}
try:
    d = np.load("phase4_nested_K4_w3200.npz", allow_pickle=True)
    ratio, label, dvth = d["chip_ratio"], d["label"], d["dvth_mv"]
    for i in range(len(ratio)):
        cat = str(label[i])
        add_row(
            sample_id=f"p4_{i}", chip_id=f"p4_{cat}_{i}",
            source_script="mc_phase4_nested.py", source_segment="phase4.2",
            generation_date=TODAY,
            physics_model="real_bsim_ptm", validation_status="simulated",
            w_um=3.2, l_um=0.045, k_pairs=4,
            pelgrom_avt_mv_um=4.5, sigma_vth_mv=11.86,
            temperature_c=27.0,  # ngspice default TNOM, no explicit .temp in this script
            dvth_intended_mv=CATEGORY_MIDPOINT_MV[cat], dvth_effective_mv=float(dvth[i]),
            aging_mechanism="natural_nbti",
            application="counterfeit_detection",
            category_label=cat, ground_truth_label=cat,
            leakage_ratio=float(ratio[i]),
        )
    print(f"Phase 4: {len(ratio)} rows processed")
except FileNotFoundError:
    print("WARNING: phase4_nested_K4_w3200.npz not found, skipping")

# --- Phase 5: tattoo uniqueness dataset (100 chips, bit=1/60mV target) ---
try:
    d = np.load("phase5_tattoo_uniqueness_K4_w3200.npz")
    i_ref, i_target = d["i_ref"], d["i_target"]
    chip_ratio = (i_target / i_ref).mean(axis=1)
    i_ref_mean_na = i_ref.mean(axis=1) * 1e9
    i_target_mean_na = i_target.mean(axis=1) * 1e9
    for i in range(len(chip_ratio)):
        add_row(
            sample_id=f"p5_auth_{i}", chip_id=f"p5_auth_{i}",
            source_script="mc_phase5_tattoo.py", source_segment="phase5.3",
            generation_date=TODAY,
            physics_model="real_bsim_ptm", validation_status="simulated",
            w_um=3.2, l_um=0.045, k_pairs=4,
            pelgrom_avt_mv_um=4.5, sigma_vth_mv=11.86,
            temperature_c=27.0,
            dvth_intended_mv=60.0, dvth_effective_mv=None,
            aging_mechanism="controlled_tattoo_stress",
            application="digital_tattoo_auth",
            category_label="tattoo_bit1", ground_truth_label="tattoo_bit1",
            tattoo_bit=1,
            i_ref_na=float(i_ref_mean_na[i]), i_target_na=float(i_target_mean_na[i]),
            leakage_ratio=float(chip_ratio[i]),
        )
    print(f"Phase 5 (auth/uniqueness): {len(chip_ratio)} rows processed")
except FileNotFoundError:
    print("WARNING: phase5_tattoo_uniqueness_K4_w3200.npz not found, skipping")

# --- Phase 5: 256-bit readback (room temp, single chip, message bits) ---
try:
    d = np.load("phase5_256bit_readback.npz")
    secret_bits, measured_ratio = d["secret_bits"], d["measured_ratio"]
    for i in range(len(secret_bits)):
        bit = int(secret_bits[i])
        add_row(
            sample_id=f"p5_256bit_{i}", chip_id="p5_256bit_chip1",
            source_script="mc_phase5_256bit.py", source_segment="phase5.2",
            generation_date=TODAY,
            physics_model="real_bsim_ptm", validation_status="simulated",
            w_um=3.2, l_um=0.045, k_pairs=4,
            pelgrom_avt_mv_um=4.5, sigma_vth_mv=11.86,
            temperature_c=27.0,
            dvth_intended_mv=60.0 if bit == 1 else 0.0,
            aging_mechanism="controlled_tattoo_stress",
            application="digital_tattoo_message",
            category_label=f"tattoo_bit{bit}", ground_truth_label=f"tattoo_bit{bit}",
            tattoo_bit=bit, bit_position=i,
            leakage_ratio=float(measured_ratio[i]),
        )
    print(f"Phase 5 (256-bit message, room temp): {len(secret_bits)} rows processed")
except FileNotFoundError:
    print("WARNING: phase5_256bit_readback.npz not found, skipping")

# --- Phase 5: CORRECTED temperature-swing dataset (fixed mismatch, same chip) ---
try:
    d = np.load("phase5_temp_ber_test_FIXED.npz")
    secret_bits = d["secret_bits"]
    ratio_calib, ratio_field = d["ratio_calib"], d["ratio_field"]
    calib_temp, field_temp = float(d["calib_temp"]), float(d["field_temp"])
    for i in range(len(secret_bits)):
        bit = int(secret_bits[i])
        for ratio_arr, temp, seg in [(ratio_calib, calib_temp, "phase5.6_calib"),
                                       (ratio_field, field_temp, "phase5.6_field")]:
            add_row(
                sample_id=f"p5_temp_{seg}_{i}", chip_id="p5_temp_fixed_chip1",
                source_script="mc_phase5_temp_ber_FIXED.py", source_segment=seg,
                generation_date=TODAY,
                physics_model="real_bsim_ptm", validation_status="simulated",
                w_um=3.2, l_um=0.045, k_pairs=4,
                pelgrom_avt_mv_um=4.5, sigma_vth_mv=11.86,
                temperature_c=temp,
                dvth_intended_mv=60.0 if bit == 1 else 0.0,
                aging_mechanism="controlled_tattoo_stress",
                application="digital_tattoo_message",
                category_label=f"tattoo_bit{bit}", ground_truth_label=f"tattoo_bit{bit}",
                tattoo_bit=bit, bit_position=i,
                leakage_ratio=float(ratio_arr[i]),
            )
    print(f"Phase 5 (temperature swing, CORRECTED, fixed-chip): {len(secret_bits)*2} rows processed")
except FileNotFoundError:
    print("WARNING: phase5_temp_ber_test_FIXED.npz not found -- run mc_phase5_temp_ber_FIXED.py first")

# --- Phase 6.3: Voltage/DIBL robustness dataset (fixed mismatch, same chip, safe VDD range) ---
try:
    d = np.load("phase6_voltage_ber_test.npz")
    secret_bits = d["secret_bits"]
    ratio_nominal, ratio_low, ratio_high = d["ratio_nominal"], d["ratio_low"], d["ratio_high"]
    vdd_nominal, vdd_low, vdd_high = float(d["vdd_nominal"]), float(d["vdd_low"]), float(d["vdd_high"])
    for i in range(len(secret_bits)):
        bit = int(secret_bits[i])
        for ratio_arr, vdd, seg in [(ratio_nominal, vdd_nominal, "phase6.3_vdd_nominal"),
                                      (ratio_low, vdd_low, "phase6.3_vdd_low"),
                                      (ratio_high, vdd_high, "phase6.3_vdd_high")]:
            add_row(
                sample_id=f"p6_vdd_{seg}_{i}", chip_id="p6_vdd_fixed_chip1",
                source_script="mc_phase6_voltage_ber.py", source_segment=seg,
                generation_date=TODAY,
                physics_model="real_bsim_ptm", validation_status="simulated",
                w_um=3.2, l_um=0.045, k_pairs=4,
                pelgrom_avt_mv_um=4.5, sigma_vth_mv=11.86,
                temperature_c=27.0, vdd_v=vdd,
                dvth_intended_mv=60.0 if bit == 1 else 0.0,
                aging_mechanism="controlled_tattoo_stress",
                application="digital_tattoo_message",
                category_label=f"tattoo_bit{bit}", ground_truth_label=f"tattoo_bit{bit}",
                tattoo_bit=bit, bit_position=i,
                leakage_ratio=float(ratio_arr[i]),
            )
    print(f"Phase 6.3 (voltage robustness, fixed-chip): {len(secret_bits)*3} rows processed")
except FileNotFoundError:
    print("WARNING: phase6_voltage_ber_test.npz not found -- run mc_phase6_voltage_ber.py first")

# --- Phase 7: 15 new independent chips for Task B chip-diversity ---
try:
    d = np.load("phase7_multichip_tattoo.npz")
    chip_idx, bit_position, bit_value, ratio = d["chip_idx"], d["bit_position"], d["bit_value"], d["ratio"]
    for i in range(len(ratio)):
        bit = int(bit_value[i])
        add_row(
            sample_id=f"p7_multichip_{chip_idx[i]}_{bit_position[i]}",
            chip_id=f"p7_multichip_{chip_idx[i]}",
            source_script="mc_phase7_multichip.py", source_segment="phase7_multichip",
            generation_date=TODAY,
            physics_model="real_bsim_ptm", validation_status="simulated",
            w_um=3.2, l_um=0.045, k_pairs=4,
            pelgrom_avt_mv_um=4.5, sigma_vth_mv=11.86,
            temperature_c=27.0,
            dvth_intended_mv=60.0 if bit == 1 else 0.0,
            aging_mechanism="controlled_tattoo_stress",
            application="digital_tattoo_message",
            category_label=f"tattoo_bit{bit}", ground_truth_label=f"tattoo_bit{bit}",
            tattoo_bit=bit, bit_position=int(bit_position[i]),
            leakage_ratio=float(ratio[i]),
        )
    print(f"Phase 7 (multichip diversity): {len(ratio)} rows processed, "
          f"{len(np.unique(chip_idx))} new independent chips")
except FileNotFoundError:
    print("WARNING: phase7_multichip_tattoo.npz not found -- run mc_phase7_multichip.py first")

# --- Report validation errors, if any ---
if errors:
    print(f"\n!!! {len(errors)} rows FAILED schema validation:")
    for sid, err in errors[:5]:
        print(f"  {sid}: {err}")
else:
    print("\nAll rows passed schema validation.")

df = pd.DataFrame(validated_rows)
df.to_csv("unified_leakage_dataset.csv", index=False)
df.to_pickle("unified_leakage_dataset.pkl")

print(f"\n=== Unified dataset saved (CSV + PKL) ===")
print(f"Total validated samples: {len(df)}")
print(f"\nBy source_segment:\n{df['source_segment'].value_counts()}")
print(f"\nBy ground_truth_label:\n{df['ground_truth_label'].value_counts()}")
print(f"\nBy temperature_c:\n{df['temperature_c'].value_counts()}")
print(f"\nBy physics_model:\n{df['physics_model'].value_counts()}")
print(f"\nUnique chip_ids: {df['chip_id'].nunique()} (use this for train/test split, not sample_id)")