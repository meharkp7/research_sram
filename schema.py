"""
Phase 6.1 -- Unified Leakage Dataset Schema (executable version)
Implements PHASE6_SCHEMA.md as an enforced pydantic model.
Any row that doesn't conform to this raises a validation error at
consolidation time, instead of silently entering the dataset.
"""

from typing import Optional, Literal
from pydantic import BaseModel, field_validator

PhysicsModel = Literal["real_bsim_ptm", "behavioral_injection", "analytical_derived"]
ValidationStatus = Literal["simulated", "analytically_extrapolated", "unvalidated_assumption"]
ProcessCorner = Literal["TT", "SS", "FF", "SF", "FS"]
AgingMechanism = Literal["natural_nbti", "controlled_tattoo_stress", "none"]
Application = Literal["counterfeit_detection", "digital_tattoo_message", "digital_tattoo_auth"]
CategoryLabel = Literal["fresh", "light", "moderate", "severe", "tattoo_bit0", "tattoo_bit1"]
SplitLabel = Literal["train", "val", "test"]


class LeakageSample(BaseModel):
    # 1. Identity & Provenance
    sample_id: str
    chip_id: str
    source_script: str
    source_segment: str
    generation_date: str
    dataset_version: str = "1.0"

    # 2. Model Validity
    physics_model: PhysicsModel
    validation_status: ValidationStatus
    silicon_validated: bool = False

    # 3. Physical Design Point
    w_um: float
    l_um: float
    k_pairs: int
    pelgrom_avt_mv_um: float
    sigma_vth_mv: float

    # 4. Environmental Conditions
    temperature_c: float
    vdd_v: Optional[float] = None
    process_corner: Optional[ProcessCorner] = None

    # 5. Aging / Stress State
    dvth_intended_mv: float
    dvth_effective_mv: Optional[float] = None
    aging_mechanism: AgingMechanism
    stress_time_years: Optional[float] = None

    # 6. Application & Ground Truth
    application: Application
    category_label: CategoryLabel
    ground_truth_label: str
    tattoo_bit: Optional[int] = None
    bit_position: Optional[int] = None  # position within a multi-bit key, where applicable

    # 7. Raw Measurement
    i_ref_na: Optional[float] = None
    i_target_na: Optional[float] = None
    leakage_ratio: float
    raw_diff_na: Optional[float] = None

    # 8. Measurement-Noise Model
    meas_noise_applied: bool = False
    meas_noise_rel_std: Optional[float] = None
    n_reads_averaged: int = 1

    # 9. Split Control
    split: Optional[SplitLabel] = None

    @field_validator("tattoo_bit")
    @classmethod
    def tattoo_bit_valid(cls, v):
        if v is not None and v not in (0, 1):
            raise ValueError("tattoo_bit must be 0, 1, or None")
        return v

    @field_validator("k_pairs", "n_reads_averaged")
    @classmethod
    def positive_int(cls, v):
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("sigma_vth_mv", "w_um", "l_um")
    @classmethod
    def positive_float(cls, v):
        if v <= 0:
            raise ValueError("must be > 0")
        return v


if __name__ == "__main__":
    # Self-test: construct one valid sample to confirm the schema works standalone
    example = LeakageSample(
        sample_id="test_0",
        chip_id="test_chip_0",
        source_script="schema.py",
        source_segment="6.1",
        generation_date="2026-07-26",
        physics_model="real_bsim_ptm",
        validation_status="simulated",
        w_um=3.2, l_um=0.045, k_pairs=4,
        pelgrom_avt_mv_um=4.5, sigma_vth_mv=11.86,
        temperature_c=27.0,
        dvth_intended_mv=0.0,
        aging_mechanism="none",
        application="counterfeit_detection",
        category_label="fresh",
        ground_truth_label="fresh",
        leakage_ratio=1.05,
    )
    print("Schema self-test passed:")
    print(example.model_dump_json(indent=2))