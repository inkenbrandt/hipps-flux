#!/usr/bin/env python3
"""
Run IRGA example on a Parquet file using ec_polars_refactored.CalcFlux

Adds robust unit handling:
- If TA looks like °C, converts to Kelvin
- If PA looks like hPa or kPa, converts to Pa
- If H2O is missing but H2O_density exists, converts density -> mixing ratio (mmol/mol)
  using ideal gas law. Autodetects units (kg/m^3 vs mol/m^3) or let user specify.

Usage:
  python run_irga_sample.py --data sample.parquet --out flux_output.parquet [--h2o-units auto|kg_m3|mol_m3]

Requires:
  pip install polars pandas numpy scipy statsmodels pyarrow
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np

R = 8.314462618  # J/(mol K)
M_W = 0.01801528 # kg/mol (water)

# Prefer Polars for reading; gracefully fall back to pandas
def read_parquet_any(path: str):
    try:
        import polars as pl
        return pl.read_parquet(path), "polars"
    except Exception:
        import pandas as pd
        try:
            df = pd.read_parquet(path)  # needs pyarrow/fastparquet
            return df, "pandas"
        except Exception as e:
            raise RuntimeError(
                f"Could not read {path}. Install one of: 'polars' OR 'pandas+pyarrow'."
            ) from e

def to_pandas(df):
    if "polars" in type(df).__module__:
        return df.to_pandas()
    return df

def detect_ts(colnames):
    cands = ["TIMESTAMP","timestamp","time","datetime","DateTime","TIMESTAMP_START"]
    for c in cands:
        if c in colnames:
            return c
    # try case-insensitive
    lower = {c.lower(): c for c in colnames}
    for c in ["timestamp","time","datetime"]:
        if c in lower:
            return lower[c]
    return None

def first_matching(colnames, patterns):
    # patterns: list of compiled regex
    for p in patterns:
        for c in colnames:
            if p.search(c):
                return c
    return None

def build_rename_map(colnames):
    # Heuristic patterns (case-insensitive)
    flags = re.IGNORECASE

    # Wind components
    ux = first_matching(colnames, [re.compile(r"^Ux$", flags), re.compile(r"^U$", flags), re.compile(r"\bu[_x]?\b", flags)])
    uy = first_matching(colnames, [re.compile(r"^Uy$", flags), re.compile(r"^V$", flags), re.compile(r"\bv[y_]?\b", flags)])
    uz = first_matching(colnames, [re.compile(r"^Uz$", flags), re.compile(r"^W$", flags), re.compile(r"\bw[z_]?\b", flags)])

    # Temperature (sonic/air)
    ta = first_matching(colnames, [
        re.compile(r"^TA$", flags),
        re.compile(r"^T[_ ]?SONIC$", flags),
        re.compile(r"^TA_1_1_1$", flags),
        re.compile(r"\bTair\b", flags),
        re.compile(r"\bAirTemp\b", flags),
        re.compile(r"^Ts$", flags),
    ])

    # Pressure
    pa = first_matching(colnames, [
        re.compile(r"^PA$", flags),
        re.compile(r"^P$|PRESS|PRESSURE", flags),
    ])

    # H2O / CO2 channels
    h2o = first_matching(colnames, [
        re.compile(r"^H2O$", flags),
        re.compile(r"H2O[_ ]?(dens|density|conc|mmol|mol)", flags),
        re.compile(r"q[_ ]?(air)?", flags),
    ])
    co2 = first_matching(colnames, [
        re.compile(r"^CO2$", flags),
        re.compile(r"CO2[_ ]?(dens|density|conc|mmol|mol)", flags),
    ])

    rename = {}
    if ux and ux != "Ux": rename[ux] = "Ux"
    if uy and uy != "Uy": rename[uy] = "Uy"
    if uz and uz != "Uz": rename[uz] = "Uz"
    if ta and ta != "TA": rename[ta] = "TA"
    if pa and pa != "PA": rename[pa] = "PA"
    if h2o and h2o != "H2O": rename[h2o] = "H2O"   # if it's already H2O, we won't overwrite later
    if co2 and co2 != "CO2": rename[co2] = "CO2"
    return rename

def coerce_temperature_K(pdf):
    if "TA" not in pdf.columns:
        return
    s = pdf["TA"].astype(float)
    # Heuristic: if range is comfortably within [-40, 60], likely °C
    if (s.dropna().between(-70, 120).all()) and (s.mean() < 200):
        pdf["TA"] = s + 273.15

def coerce_pressure_Pa(pdf):
    if "PA" not in pdf.columns:
        return
    s = pdf["PA"].astype(float)
    m = float(np.nanmean(s))
    # Heuristics:
    #   ~1013  => hPa -> Pa
    #   ~101   => kPa -> Pa
    #   ~ 1    => MPa -> Pa (unlikely for ambient)
    if 300 <= m <= 1300:        # hPa
        pdf["PA"] = s * 100.0
    elif 50 <= m <= 200:        # kPa
        pdf["PA"] = s * 1000.0
    elif 0.5 <= m <= 5.0:       # MPa (unlikely)
        pdf["PA"] = s * 1_000_000.0
    # else assume already Pa (~101325)

def convert_h2o_density_to_mmr(pdf, units: str = "auto"):
    """
    Convert H2O_density -> H2O (mmol/mol) using ideal gas law.
    Requires TA [K] and PA [Pa].
      - If units='kg_m3': H2O_density is kg/m^3
      - If units='mol_m3': H2O_density is mol/m^3
      - If units='auto': choose based on magnitude
    """
    dens_col = None
    for c in pdf.columns:
        cl = c.lower()
        if "h2o" in cl and ("dens" in cl or "density" in cl):
            dens_col = c
            break
    if dens_col is None:
        return  # nothing to do
    if "H2O" in pdf.columns:
        return  # already present

    if "TA" not in pdf.columns or "PA" not in pdf.columns:
        raise ValueError("Need TA (K) and PA (Pa) to convert H2O_density to mixing ratio.")

    T = pdf["TA"].astype(float).to_numpy()
    P = pdf["PA"].astype(float).to_numpy()
    rho = pdf[dens_col].astype(float).to_numpy()

    # Autodetect units by typical magnitude
    sel_units = units
    mean_rho = float(np.nanmean(rho))
    if units == "auto":
        # Saturation water vapor density near 20C ~ 0.017 kg/m^3; typical molar density order ~ 1 mol/m^3
        if mean_rho < 0.2:      # looks like kg/m^3
            sel_units = "kg_m3"
        elif mean_rho < 50:     # 0.2 .. 50 mol/m^3 plausible for moist air range
            sel_units = "mol_m3"
        else:
            # already in mmol/mol? then just copy across as a last resort
            pdf["H2O"] = pdf[dens_col].astype(float)
            return

    if sel_units == "kg_m3":
        n_w = rho / M_W                  # mol/m^3
    elif sel_units == "mol_m3":
        n_w = rho                         # mol/m^3
    else:
        raise ValueError("units must be one of: auto|kg_m3|mol_m3")

    # Total molar density n_total = P/(R T)  [mol/m^3]
    n_tot = P / (R * T)
    # Mixing ratio (mol/mol)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_molmol = n_w / n_tot
    # Convert to mmol/mol
    pdf["H2O"] = q_molmol * 1e3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("sample.parquet"))
    p.add_argument("--out", type=Path, default=Path("flux_output.parquet"))
    p.add_argument("--h2o-units", choices=["auto","kg_m3","mol_m3"], default="auto",
                   help="Units for H2O_density if present (auto-detect by default).")
    args = p.parse_args()

    df, engine = read_parquet_any(str(args.data))
    print(f"Loaded {args.data} with {engine}. shape=", getattr(df, "shape", None))

    # Column list
    colnames = df.columns if engine == "polars" else list(df.columns)
    print("Detected columns (first 25):", colnames[:25])

    # Timestamp detection
    ts_col = detect_ts(colnames)
    if ts_col:
        print(f"Using timestamp column: {ts_col}")

    # Build rename map heuristically
    rename_map = build_rename_map(colnames)
    if rename_map:
        print("Proposed rename_map:", rename_map)
    else:
        print("No renaming needed or could not detect differences.")

    # Convert to pandas for preprocessing and for run_irga compatibility
    pdf = to_pandas(df)

    # Apply renames early so unit conversions see canonical columns
    if rename_map:
        pdf = pdf.rename(columns=rename_map)

    # Unit coercions
    if "TA" in pdf.columns:
        coerce_temperature_K(pdf)
    if "PA" in pdf.columns:
        coerce_pressure_Pa(pdf)

    # H2O from density if needed
    try:
        convert_h2o_density_to_mmr(pdf, units=args.h2o_units)
    except ValueError as e:
        print("H2O conversion skipped:", e)

    # Import processor
    from ec_polars_refactored import CalcFlux
    fx = CalcFlux()

    # Run
    print("Running run_irga...")
    series = fx.run_irga(pdf, rename_map=None, ts_col=ts_col)  # already renamed & coerced
    print("Output series (head):")
    try:
        print(series.head())
    except Exception:
        print(series)

    # Try to collect key outputs into a one-row DataFrame and save
    try:
        import pandas as pd
        out_df = pd.DataFrame([series.to_dict()])
        out_df.to_parquet(args.out, index=False)
        print(f"Wrote {args.out}")
    except Exception as e:
        print("Could not write Parquet, falling back to CSV:", e)
        out_csv = args.out.with_suffix(".csv")
        try:
            out_df.to_csv(out_csv, index=False)
            print(f"Wrote {out_csv}")
        except Exception as e2:
            print("Failed to write CSV as well:", e2)

    print("Done.")

if __name__ == "__main__":
    main()
