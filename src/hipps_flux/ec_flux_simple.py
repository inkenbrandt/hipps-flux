
"""
Lightweight eddy-covariance processor that works with the provided sample data.

- Accepts a pandas DataFrame (or CSV/Parquet path).
- Expects columns similar to: Ux, Uy, Uz, T_SONIC_corr (or T_SONIC), PA (kPa), H2O_density (g m-3).
- Computes basic flux stats: u*, mean wind, H, LE, ET, covariances, std devs.
- Uses a fast MAD-based despiker.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

R_v = 461.5       # J kg-1 K-1, gas constant of water vapor
R_d = 287.05      # J kg-1 K-1, gas constant of dry air
Cp_d = 1004.7     # J kg-1 K-1
rho_w = 1000.0    # kg m-3
Lv = 2.45e6       # J kg-1

def _to_df(df_or_path) -> pd.DataFrame:
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()
    if isinstance(df_or_path, str):
        if df_or_path.lower().endswith(".parquet"):
            try:
                return pd.read_parquet(df_or_path)
            except Exception:
                pass
        return pd.read_csv(df_or_path)
    raise TypeError("Provide a pandas DataFrame or a path to CSV/Parquet.")

def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "T_SONIC_corr": "Ts",
        "T_SONIC": "Ts",
        "TA_1_1_1": "Ta",
        "PA": "Pr",                 # kPa
        "amb_press": "Pr",
        "H2O_density": "pV",        # g m-3
        "CO2_density": "CO2"
    })

def _mad_mask(x: np.ndarray, win: int = 201, k: float = 3.0) -> np.ndarray:
    """Return boolean mask of spikes using rolling median residual MAD."""
    s = pd.Series(x)
    med = s.rolling(win, center=True, min_periods=1).median()
    resid = s - med
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid))) + 1e-12
    return np.abs(resid.values) > (k * 1.4826 * mad)

def _despike(x: np.ndarray, win: int = 201) -> np.ndarray:
    mask = _mad_mask(x, win=win)
    y = pd.Series(x).mask(mask).interpolate().bfill().ffill().values
    return y

def _cov(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x); y = np.asarray(y)
    x = x - np.nanmean(x); y = y - np.nanmean(y)
    return float(np.nanmean(x * y))

class CalcFluxSimple:
    def __init__(self, fast: bool = True, despike_win: int = 201):
        self.fast = fast
        self.despike_win = despike_win

    def run_irga(self, df_or_path) -> pd.Series:
        df = _to_df(df_or_path)
        df = _rename_cols(df)

        needed = ["Ux","Uy","Uz","Ts","Pr","pV"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Despike core variables
        for c in ["Ux","Uy","Uz","Ts","pV","Pr"]:
            if c in df.columns:
                df[c] = _despike(df[c].to_numpy(), win=self.despike_win)

        # Units: Ts in °C -> K; Pr in kPa -> Pa; pV g m-3 -> kg m-3
        T_sonic_K = df["Ts"].to_numpy() + 273.15
        P_pa = df["Pr"].to_numpy() * 1000.0
        rho_v = df["pV"].to_numpy() * 1e-3  # kg m-3

        # Air density approx (moist): rho = (P - e)/ (R_d*T) + e/(R_v*T)
        # Here estimate e from rho_v and T: e = rho_v * R_v * T
        e_pa = rho_v * R_v * T_sonic_K
        rho_air = (P_pa - e_pa) / (R_d * T_sonic_K) + (e_pa) / (R_v * T_sonic_K)

        Ux = df["Ux"].to_numpy(); Uy = df["Uy"].to_numpy(); Uz = df["Uz"].to_numpy()
        U = np.sqrt(Ux**2 + Uy**2)

        # Friction velocity u* = (cov(Ux',Uz')^2 + cov(Uy',Uz')^2)^(1/4)
        uw = _cov(Ux, Uz); vw = _cov(Uy, Uz)
        ustar = np.sqrt(np.sqrt(uw**2 + vw**2))

        # Sensible heat H ≈ rho * Cp * cov(T, w), with T in K
        H = float(np.nanmean(rho_air) * Cp_d * _cov(T_sonic_K, Uz))

        # Latent heat from water density (approx): convert rho_v (kg m-3) -> q' via q ≈ 0.622 e/(P - 0.378 e)
        # Simpler: use water density fluctuations to estimate LE ~ Lv * cov(q, w) * rho_air (very rough)
        # We'll proxy q' ≈ rho_v / rho_air
        q = rho_v / np.maximum(rho_air, 1e-6)
        LE = float(np.nanmean(rho_air) * Lv * _cov(q, Uz))

        # ET (mm d-1) from LE: LE (W m-2) / (rho_w * Lv) -> kg m-2 s-1, convert to mm d-1
        ET_mm_d = float((LE / (rho_w * Lv)) * 86400.0 * 1000.0)

        out = pd.Series({
            "Ta": float(np.nanmean(df.get("Ta", df["Ts"]))),   # °C
            "Ustr": float(ustar),
            "Uxy": float(np.nanmean(U)),
            "H": float(H),
            "lambdaE": float(LE),
            "ET": float(ET_mm_d),
            "StDevUz": float(np.nanstd(Uz)),
            "StDevTa": float(np.nanstd(T_sonic_K)),
        })
        return out
