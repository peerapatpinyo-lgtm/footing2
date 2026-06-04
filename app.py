import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass, field
from typing import List, Optional
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CORE DATA STRUCTURES
# ==========================================
@dataclass
class Loads:
    P_DL: float; P_LL: float
    M_DL_x: float; M_LL_x: float; M_WL_x: float
    M_DL_y: float; M_LL_y: float; M_WL_y: float
    V_hx: float; V_hy: float

@dataclass
class Properties:
    qa_allow: float; fc_prime: float; fy: float
    soil_density: float; base_friction: float

@dataclass
class SoilProfile:
    """Geotechnical parameters for bearing capacity & settlement"""
    # Shear strength
    cohesion: float = 0.0          # c  (ton/m²)
    phi_deg: float  = 30.0         # φ  (degrees)
    # Settlement – elastic
    Es: float       = 2000.0       # Modulus of elasticity (ton/m²)
    nu: float       = 0.3          # Poisson's ratio
    # Settlement – consolidation
    Cc: float       = 0.3          # Compression index
    Cs: float       = 0.05         # Swelling index
    e0: float       = 0.8          # Initial void ratio
    OCR: float      = 1.0          # Over-consolidation ratio
    H_clay: float   = 3.0          # Clay layer thickness (m)
    sigma_v0: float = 10.0         # Initial effective vertical stress at mid-layer (ton/m²)
    soil_density: float = 1.8     # Unit weight of soil (ton/m³)
    # Bearing capacity method
    method: str     = "Meyerhof"   # "Terzaghi" or "Meyerhof"
    # Failure mode (Terzaghi only)
    failure_mode: str = "General"  # "General", "Local", "Punching"

@dataclass
class Geometry:
    B: float; L: float; H_cm: float; Df: float
    cx: float; cy: float
    shape: str = "rectangular"   # "rectangular" or "circular"
    D_circ: float = 0.0          # diameter if circular (m)

# ==========================================
# 2. ENGINEERING CALCULATION ENGINE (OOP)
# ==========================================
class FoundationDesigner:
    def __init__(self, loads: Loads, props: Properties, geo: Geometry):
        self.loads = loads
        self.props = props
        self.geo = geo

        self.is_circular = (geo.shape == "circular")

        if self.is_circular:
            # Treat circle as equivalent square for internal calcs
            R = geo.D_circ / 2
            self.A_base = math.pi * R ** 2
            self.B_eff = math.sqrt(self.A_base)   # equivalent side for shear/moment
            self.L_eff = self.B_eff
            self.B_cm = self.B_eff * 100
            self.L_cm = self.B_eff * 100
            self.I_x = math.pi * (geo.D_circ ** 4) / 64
            self.I_y = self.I_x
        else:
            self.A_base = geo.B * geo.L
            self.B_cm = geo.B * 100
            self.L_cm = geo.L * 100
            self.I_x = (geo.B * geo.L ** 3) / 12
            self.I_y = (geo.L * geo.B ** 3) / 12

        self.d_cm = geo.H_cm - 7.5   # Effective depth (cm)

    # ----------------------------------------------------------
    # SERVICE LIMIT STATE
    # ----------------------------------------------------------
    def analyze_service_state(self):
        P_service = self.loads.P_DL + self.loads.P_LL
        M_service_x = self.loads.M_DL_x + self.loads.M_LL_x
        M_service_y = self.loads.M_DL_y + self.loads.M_LL_y

        H_m = self.geo.H_cm / 100
        W_footing = self.A_base * H_m * 2.4
        if self.is_circular:
            Df_soil = max(0, self.geo.Df - H_m)
            W_overburden = (self.geo.Df * self.A_base - H_m * self.A_base) * self.props.soil_density
        else:
            W_overburden = self.A_base * (self.geo.Df - H_m) * self.props.soil_density
        P_total = P_service + W_footing + W_overburden

        if self.is_circular:
            R = self.geo.D_circ / 2
            kern = R / 4          # kern for circle = R/4
            kern_x = kern_y = kern
        else:
            kern_x, kern_y = self.geo.B / 6, self.geo.L / 6

        e_x = M_service_y / P_total if P_total > 0 else 0
        e_y = M_service_x / P_total if P_total > 0 else 0

        has_tension = (e_x > kern_x) or (e_y > kern_y)

        q_avg = P_total / self.A_base

        if self.is_circular:
            R = self.geo.D_circ / 2
            # Section modulus for circle
            S_x = math.pi * R ** 3 / 4
            S_y = S_x
            q_max = q_avg + M_service_x / S_x + M_service_y / S_y
            q_min = max(0.0, q_avg - M_service_x / S_x - M_service_y / S_y)
        else:
            q_mod_x = (M_service_y * (self.geo.B / 2)) / self.I_y
            q_mod_y = (M_service_x * (self.geo.L / 2)) / self.I_x
            if not has_tension:
                q_max = q_avg + q_mod_x + q_mod_y
                q_min = max(0.0, q_avg - q_mod_x - q_mod_y)
            else:
                B_prime = max(self.geo.B - 2 * e_x, 0.1)
                L_prime = max(self.geo.L - 2 * e_y, 0.1)
                factor = 4.0 / 3.0 if (e_x > kern_x and e_y > kern_y) else 1.0
                q_max = P_total / (B_prime * L_prime) * factor
                q_min = 0.0

        # Overturning & Sliding
        if self.is_circular:
            half_x = half_y = self.geo.D_circ / 2
        else:
            half_x = self.geo.B / 2
            half_y = self.geo.L / 2

        M_res_x = P_total * half_y
        M_ovr_x = M_service_x + (self.loads.V_hy * self.geo.Df)
        FS_ovr_x = M_res_x / M_ovr_x if M_ovr_x > 0 else float('inf')

        M_res_y = P_total * half_x
        M_ovr_y = M_service_y + (self.loads.V_hx * self.geo.Df)
        FS_ovr_y = M_res_y / M_ovr_y if M_ovr_y > 0 else float('inf')

        V_h_total = math.sqrt(self.loads.V_hx ** 2 + self.loads.V_hy ** 2)
        FS_slide = (P_total * self.props.base_friction) / V_h_total if V_h_total > 0 else float('inf')

        return {
            "P_total": P_total, "e_x": e_x, "e_y": e_y,
            "kern_x": kern_x, "kern_y": kern_y, "has_tension": has_tension,
            "q_max": q_max, "q_min": q_min,
            "FS_ovr_x": FS_ovr_x, "FS_ovr_y": FS_ovr_y, "FS_slide": FS_slide
        }

    # ----------------------------------------------------------
    # ULTIMATE LIMIT STATE  (all in ksc / ton-cm units)
    # Fixed: separate per-direction qu, proper load combos
    # ----------------------------------------------------------
    def _factored_loads(self):
        """Return governing factored loads per ACI 318-19 load combos.
        Returns dict with P_u(ton), M_u_x(ton-m), M_u_y(ton-m)
        """
        combos = [
            # 1.4D
            dict(
                P=1.4 * self.loads.P_DL,
                Mx=1.4 * self.loads.M_DL_x,
                My=1.4 * self.loads.M_DL_y,
                label="1.4D"
            ),
            # 1.2D + 1.6L
            dict(
                P=1.2 * self.loads.P_DL + 1.6 * self.loads.P_LL,
                Mx=1.2 * self.loads.M_DL_x + 1.6 * self.loads.M_LL_x,
                My=1.2 * self.loads.M_DL_y + 1.6 * self.loads.M_LL_y,
                label="1.2D+1.6L"
            ),
            # 1.2D + 1.0L + 1.0W
            dict(
                P=1.2 * self.loads.P_DL + 1.0 * self.loads.P_LL,
                Mx=1.2 * self.loads.M_DL_x + 1.0 * self.loads.M_LL_x + 1.0 * self.loads.M_WL_x,
                My=1.2 * self.loads.M_DL_y + 1.0 * self.loads.M_LL_y + 1.0 * self.loads.M_WL_y,
                label="1.2D+1.0L+1.0W"
            ),
            # 0.9D + 1.0W  (uplift check)
            dict(
                P=0.9 * self.loads.P_DL,
                Mx=0.9 * self.loads.M_DL_x + 1.0 * self.loads.M_WL_x,
                My=0.9 * self.loads.M_DL_y + 1.0 * self.loads.M_WL_y,
                label="0.9D+1.0W"
            ),
        ]
        # Governing = max total qu at worst corner
        best = max(combos, key=lambda c: c["P"] + abs(c["Mx"]) + abs(c["My"]))
        return best

    def analyze_ultimate_state(self):
        fc = self.props.fc_prime  # ksc

        combo = self._factored_loads()
        P_u   = combo["P"]        # tons
        M_u_x = combo["Mx"]       # ton-m
        M_u_y = combo["My"]       # ton-m
        governing_combo = combo["label"]

        # Convert to ksc on cm² base: 1 ton = 1000 kg, area in cm²
        A_cm2 = self.B_cm * self.L_cm  # cm² (equivalent for circular too)

        qu_base  = (P_u * 1000) / A_cm2                                          # kg/cm²
        # qu per direction: x-bending creates gradient along B, y-bending along L
        # M_u_y causes gradient in x-direction (across B)
        qu_mod_x = (M_u_y * 1e5 * (self.B_cm / 2)) / ((self.L_cm * self.B_cm ** 3) / 12)  # kg/cm²
        # M_u_x causes gradient in y-direction (across L)
        qu_mod_y = (M_u_x * 1e5 * (self.L_cm / 2)) / ((self.B_cm * self.L_cm ** 3) / 12)  # kg/cm²

        # Store per-corner pressures
        qu_corners = {
            "max_x": qu_base + qu_mod_x,
            "max_y": qu_base + qu_mod_y,
            "max_both": qu_base + qu_mod_x + qu_mod_y,
            "min_both": max(0.0, qu_base - qu_mod_x - qu_mod_y),
        }
        qu_max = qu_corners["max_both"]

        # ── One-Way Shear ─────────────────────────────────────────────
        # X-direction: critical section at d from column face, strip width = L_cm
        crit_x = max(0.0, ((self.B_cm - self.geo.cx) / 2) - self.d_cm)
        # Use average pressure at mid-cantilever for X strip
        qu_x_avg = (qu_base + qu_mod_x)  # conservative: max in that direction
        V_u_x = qu_x_avg * self.L_cm * crit_x   # kg  (force on strip)
        v_u_wide_x = V_u_x / (self.L_cm * self.d_cm)  # kg/cm²

        # Y-direction: critical section at d from column face, strip width = B_cm
        crit_y = max(0.0, ((self.L_cm - self.geo.cy) / 2) - self.d_cm)
        qu_y_avg = (qu_base + qu_mod_y)  # conservative: max in that direction
        V_u_y = qu_y_avg * self.B_cm * crit_y   # kg
        v_u_wide_y = V_u_y / (self.B_cm * self.d_cm)  # kg/cm²

        phi_v_c_wide = 0.75 * 0.53 * math.sqrt(fc)  # kg/cm²

        # ── Punching Shear ────────────────────────────────────────────
        bo = 2 * ((self.geo.cx + self.d_cm) + (self.geo.cy + self.d_cm))
        area_punch = (self.geo.cx + self.d_cm) * (self.geo.cy + self.d_cm)
        # Punching load uses max qu
        V_punch = qu_max * (A_cm2 - area_punch)   # kg
        v_u_punch = V_punch / (bo * self.d_cm)    # kg/cm²

        beta_c = max(self.geo.cx, self.geo.cy) / min(self.geo.cx, self.geo.cy)
        vc1 = 0.27 * (2 + 4 / beta_c) * math.sqrt(fc)
        vc2 = 0.27 * ((40 * self.d_cm / bo) + 2) * math.sqrt(fc)
        vc3 = 1.06 * math.sqrt(fc)
        phi_v_c_punch = 0.75 * min(vc1, vc2, vc3)

        # ── Flexure ─── per-direction using correct qu ────────────────
        cant_x = (self.B_cm - self.geo.cx) / 2
        cant_y = (self.L_cm - self.geo.cy) / 2
        # Moment about face of column (kg·cm)
        M_ux = (qu_x_avg * self.L_cm * cant_x ** 2) / 2   # kg·cm per cm width × L_cm
        M_uy = (qu_y_avg * self.B_cm * cant_y ** 2) / 2   # kg·cm

        # ── Development Length ────────────────────────────────────────
        db_size = 16  # mm  (DB16)
        L_d = (self.props.fy / (1.4 * math.sqrt(fc))) * (db_size / 10)
        available_L_d_x = cant_x - 7.5
        available_L_d_y = cant_y - 7.5

        return {
            "governing_combo": governing_combo,
            "P_u": P_u, "M_u_x": M_u_x, "M_u_y": M_u_y,
            "qu_base": qu_base, "qu_max": qu_max, "qu_corners": qu_corners,
            "v_u_wide_x": v_u_wide_x, "v_u_wide_y": v_u_wide_y,
            "v_u_wide_max": max(v_u_wide_x, v_u_wide_y),
            "phi_v_c_wide": phi_v_c_wide,
            "v_u_punch": v_u_punch, "phi_v_c_punch": phi_v_c_punch,
            "M_ux": M_ux, "M_uy": M_uy, "bo": bo,
            "L_d": L_d,
            "available_L_d_x": available_L_d_x, "available_L_d_y": available_L_d_y,
        }

    def design_flexure(self, M_u, width_cm):
        """M_u in kg·cm, width_cm in cm → returns (bars_count, spacing_cm)"""
        rho_min = 0.0018 if self.props.fy >= 4000 else 0.0020
        d = self.d_cm
        fc = self.props.fc_prime
        fy = self.props.fy

        R_n = M_u / (0.90 * width_cm * d ** 2)   # kg/cm²
        m = fy / (0.85 * fc)
        discriminant = 1 - (2 * m * R_n / fy)
        rho_req = (1 / m) * (1 - math.sqrt(max(0, discriminant))) if discriminant > 0 else rho_min

        As_req = max(rho_req, rho_min) * width_cm * d    # cm²
        bars_req = max(5, math.ceil(As_req / 2.01))      # DB16 = 2.01 cm²
        spacing = (width_cm - 15) / max(bars_req - 1, 1)
        return bars_req, spacing

    # ----------------------------------------------------------
    # SOIL PRESSURE GRID (for 3-D plot)
    # ----------------------------------------------------------
    def soil_pressure_grid(self, n=40):
        sls = self.analyze_service_state()
        P_total = sls["P_total"]
        q_avg = P_total / self.A_base

        if self.is_circular:
            R = self.geo.D_circ / 2
            xs = np.linspace(-R, R, n)
            ys = np.linspace(-R, R, n)
            XX, YY = np.meshgrid(xs, ys)
            S = math.pi * R ** 3 / 4
            ZZ = q_avg + (self.loads.M_DL_x + self.loads.M_LL_x) / S * YY / R + \
                          (self.loads.M_DL_y + self.loads.M_LL_y) / S * XX / R
            mask = XX ** 2 + YY ** 2 > R ** 2
            ZZ[mask] = np.nan
        else:
            B, L = self.geo.B, self.geo.L
            xs = np.linspace(-B / 2, B / 2, n)
            ys = np.linspace(-L / 2, L / 2, n)
            XX, YY = np.meshgrid(xs, ys)
            M_x = self.loads.M_DL_x + self.loads.M_LL_x
            M_y = self.loads.M_DL_y + self.loads.M_LL_y
            ZZ = (q_avg
                  + M_y * XX / self.I_y
                  + M_x * YY / self.I_x)
            ZZ = np.maximum(ZZ, 0)
        return XX, YY, ZZ

    # ----------------------------------------------------------
    # AUTO-OPTIMIZATION: find smallest B×L passing all checks
    # ----------------------------------------------------------
    def optimize_dimensions(self, aspect_ratio: float = 1.0):
        """
        Find minimum B (and L = B/aspect_ratio if aspect_ratio≠1) that passes:
         - q_max ≤ qa_allow
         - FS_ovr_x ≥ 1.5, FS_ovr_y ≥ 1.5, FS_slide ≥ 1.5
         - v_u_wide ≤ phi_v_c, v_u_punch ≤ phi_v_c_punch
        Keeps H, Df, cx, cy, shape unchanged.
        """
        import copy

        def violation(B_try):
            g2 = copy.copy(self.geo)
            g2.B = round(B_try, 2)
            g2.L = round(B_try / aspect_ratio, 2)
            des = FoundationDesigner(self.loads, self.props, g2)
            sls = des.analyze_service_state()
            uls = des.analyze_ultimate_state()

            # Margin: positive = good (capacity > demand)
            geo_margin = min(
                self.props.qa_allow - sls["q_max"],
                sls["FS_ovr_x"] - 1.5,
                sls["FS_ovr_y"] - 1.5,
                sls["FS_slide"] - 1.5,
            )
            str_margin = min(
                uls["phi_v_c_wide"] - uls["v_u_wide_max"],
                uls["phi_v_c_punch"] - uls["v_u_punch"],
            )
            return -min(geo_margin, str_margin)  # minimise: want this ≤ 0

        # Binary search between 0.5 m and 10 m
        lo, hi = 0.5, 10.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if violation(mid) <= 0:
                hi = mid
            else:
                lo = mid

        B_opt = math.ceil(hi * 10) / 10  # round up to nearest 0.1 m
        L_opt = math.ceil((B_opt / aspect_ratio) * 10) / 10
        return B_opt, L_opt


# ==========================================
# 3. BEARING CAPACITY ENGINE
# ==========================================
class BearingCapacityEngine:
    """
    Computes ultimate & allowable bearing capacity.
    Supports Terzaghi (General / Local / Punching) and Meyerhof
    (with shape, depth, inclination factors).
    """

    def __init__(self, soil: SoilProfile, geo: Geometry, loads: Loads):
        self.soil  = soil
        self.geo   = geo
        self.loads = loads
        self.phi   = math.radians(soil.phi_deg)

    # ── Terzaghi Bearing Capacity Factors ───────────────────────
    def _terzaghi_factors(self, phi_use):
        """Nq, Nc, Nγ for given phi (radians)."""
        if phi_use == 0:
            Nq = 1.0; Nc = 5.7; Ng = 0.0
        else:
            Nq = math.exp(math.pi * math.tan(phi_use)) * (math.tan(math.radians(45) + phi_use / 2)) ** 2
            Nc = (Nq - 1) / math.tan(phi_use)
            Ng = 2 * (Nq + 1) * math.tan(phi_use)
        return Nq, Nc, Ng

    def terzaghi(self):
        c   = self.soil.cohesion
        phi = self.phi
        γ   = self.props_gamma()
        Df  = self.geo.Df
        B   = self.geo.B if not (self.geo.shape == "circular") else self.geo.D_circ

        mode = self.soil.failure_mode
        phi_use = phi
        c_use   = c

        if mode == "Local":
            phi_use = math.atan(0.667 * math.tan(phi))
            c_use   = 0.667 * c
        elif mode == "Punching":
            phi_use = math.atan(0.5 * math.tan(phi))
            c_use   = 0.5 * c

        Nq, Nc, Ng = self._terzaghi_factors(phi_use)

        is_circ  = (self.geo.shape == "circular")
        is_strip = False  # always spread footing here

        if is_circ:
            qu = 1.3 * c_use * Nc + γ * Df * Nq + 0.3 * γ * B * Ng
        else:
            # Rectangular shape factors (Terzaghi)
            B_L = B / self.geo.L
            sc = 1 + 0.3 * B_L
            sq = 1 + B_L * math.tan(phi_use) if phi_use > 0 else 1.0
            sg = max(1 - 0.4 * B_L, 0.6)
            qu = c_use * Nc * sc + γ * Df * Nq * sq + 0.5 * γ * B * Ng * sg

        FS_bearing = 3.0
        qa_computed = qu / FS_bearing

        return {
            "method": f"Terzaghi ({mode})",
            "phi_use_deg": math.degrees(phi_use),
            "c_use": c_use,
            "Nq": Nq, "Nc": Nc, "Ng": Ng,
            "qu_ultimate": qu,
            "FS_bearing": FS_bearing,
            "qa_computed": qa_computed,
        }

    # ── Meyerhof Bearing Capacity Factors ───────────────────────
    def _meyerhof_factors(self):
        phi = self.phi
        if phi == 0:
            Nq = 1.0; Nc = 5.14; Ng = 0.0
        else:
            Nq = math.exp(math.pi * math.tan(phi)) * (math.tan(math.radians(45) + phi / 2)) ** 2
            Nc = (Nq - 1) / math.tan(phi)
            Ng = (Nq - 1) * math.tan(1.4 * phi)
        return Nq, Nc, Ng

    def meyerhof(self):
        c   = self.soil.cohesion
        phi = self.phi
        γ   = self.props_gamma()
        Df  = self.geo.Df
        is_circ = (self.geo.shape == "circular")
        B = self.geo.D_circ if is_circ else self.geo.B
        L = B if is_circ else self.geo.L

        Nq, Nc, Ng = self._meyerhof_factors()

        # Shape factors
        if phi == 0:
            sc = 1 + 0.2 * (B / L)
            sq = sg = 1.0
        else:
            sc = 1 + 0.2 * (B / L) * math.tan(math.radians(45) + phi / 2) ** 2
            sq = sg = 1 + 0.1 * (B / L) * math.tan(math.radians(45) + phi / 2) ** 2

        # Depth factors
        Df_B = Df / B
        if phi == 0:
            dc = 1 + 0.4 * Df_B
            dq = dg = 1.0
        else:
            dc = 1 + 0.4 * Df_B
            dq = dg = 1 + 0.1 * Df_B * math.tan(math.radians(45) + phi / 2) ** 2

        # Inclination factors (resultant horizontal load)
        P_service = self.loads.P_DL + self.loads.P_LL
        V_h = math.sqrt(self.loads.V_hx ** 2 + self.loads.V_hy ** 2)
        alpha_deg = math.degrees(math.atan(V_h / P_service)) if P_service > 0 else 0
        if phi == 0:
            ic = 1 - alpha_deg / 90
            iq = ig = 1.0
        else:
            ic = iq = (1 - alpha_deg / 90) ** 2
            ig = (1 - alpha_deg / phi) ** 2 if phi > 0 else 1.0

        qu = (c * Nc * sc * dc * ic
              + γ * Df * Nq * sq * dq * iq
              + 0.5 * γ * B * Ng * sg * dg * ig)

        FS_bearing = 3.0
        qa_computed = qu / FS_bearing

        return {
            "method": "Meyerhof",
            "Nq": Nq, "Nc": Nc, "Ng": Ng,
            "sc": sc, "sq": sq, "sg": sg,
            "dc": dc, "dq": dq, "dg": dg,
            "ic": ic, "iq": iq, "ig": ig,
            "alpha_deg": alpha_deg,
            "qu_ultimate": qu,
            "FS_bearing": FS_bearing,
            "qa_computed": qa_computed,
        }

    def props_gamma(self):
        return self.soil.soil_density if hasattr(self.soil, 'soil_density') else 1.8

    def run(self):
        if self.soil.method == "Terzaghi":
            return self.terzaghi()
        else:
            return self.meyerhof()

    def run_both(self):
        return {"Terzaghi": self.terzaghi(), "Meyerhof": self.meyerhof()}


# ==========================================
# 4. SETTLEMENT ENGINE
# ==========================================
class SettlementEngine:
    """
    Elastic (Schleicher) + Primary Consolidation settlement.
    Units: tons, metres.
    """

    def __init__(self, soil: SoilProfile, geo: Geometry, q_net: float):
        self.soil  = soil
        self.geo   = geo
        self.q_net = q_net   # net foundation pressure (ton/m²)

    # ── Elastic Settlement ───────────────────────────────────────
    def elastic(self):
        """
        Si = q_net · B · (1-ν²) / Es · If
        If = influence factor (Bowles, ~0.82 for flexible square, ~0.54 rigid)
        """
        B  = self.geo.D_circ if self.geo.shape == "circular" else self.geo.B
        L  = B if self.geo.shape == "circular" else self.geo.L
        Es = self.soil.Es
        nu = self.soil.nu

        # Influence factor (Steinbrenner approximation for L/B)
        LB = L / B
        If = 0.82 * (1 + 0.22 * (LB - 1))  # flexible, centre point

        Si_m = self.q_net * B * (1 - nu ** 2) / Es * If
        Si_cm = Si_m * 100
        return {"Si_m": Si_m, "Si_cm": Si_cm, "If": If}

    # ── Consolidation Settlement ─────────────────────────────────
    def consolidation(self):
        """
        Sc = Cc·H/(1+e0)·log10[(σ'v0 + Δσ) / σ'v0]   (NC clay)
        Sc = Cs·H/(1+e0)·log10[(σ'v0 + Δσ) / σ'vc]
           + Cc·H/(1+e0)·log10[(σ'vc + Δσ) / σ'vc]    (OC clay, if stress exceeds precon.)
        Δσ estimated via Boussinesq 2:1 distribution at mid-layer.
        """
        Cc   = self.soil.Cc
        Cs   = self.soil.Cs
        e0   = self.soil.e0
        OCR  = self.soil.OCR
        H    = self.soil.H_clay
        sv0  = self.soil.sigma_v0
        svc  = sv0 * OCR          # preconsolidation stress

        B   = self.geo.D_circ if self.geo.shape == "circular" else self.geo.B
        L   = B if self.geo.shape == "circular" else self.geo.L
        Df  = self.geo.Df

        # Boussinesq 2:1 stress increment at mid-layer depth below footing base
        z   = H / 2
        dsigma = self.q_net * B * L / ((B + z) * (L + z))

        sv1 = sv0 + dsigma   # final stress

        if OCR <= 1.0 or sv0 >= svc:  # NC clay
            if sv0 > 0:
                Sc = (Cc * H / (1 + e0)) * math.log10(sv1 / sv0)
            else:
                Sc = 0.0
            regime = "Normally Consolidated (NC)"
        else:
            if sv1 <= svc:             # OC – stays in recompression
                Sc = (Cs * H / (1 + e0)) * math.log10(sv1 / sv0)
                regime = "Over-Consolidated (OC) – recompression only"
            else:                       # OC – crosses precon. pressure
                Sc = ((Cs * H / (1 + e0)) * math.log10(svc / sv0)
                    + (Cc * H / (1 + e0)) * math.log10(sv1 / svc))
                regime = "Over-Consolidated (OC) – virgin compression"

        Sc_cm = Sc * 100
        return {
            "Sc_m": Sc, "Sc_cm": Sc_cm,
            "dsigma": dsigma, "sv0": sv0, "svc": svc, "sv1": sv1,
            "regime": regime,
        }

    def total(self):
        el  = self.elastic()
        con = self.consolidation()
        St  = el["Si_m"] + con["Sc_m"]
        return {
            "elastic": el,
            "consolidation": con,
            "St_m": St,
            "St_cm": St * 100,
        }


# ==========================================
# 5. COMBINED FOOTING DESIGNER
# ==========================================
@dataclass
class ColumnData:
    P_DL: float; P_LL: float          # tons
    cx: float;   cy: float            # cm
    x_pos: float                       # position along footing length (m from left edge)

class CombinedFootingDesigner:
    """
    Rectangular or Trapezoidal combined footing for 2 columns.
    Columns are aligned along the L (length) axis.
    """

    def __init__(self, col1: ColumnData, col2: ColumnData,
                 B: float, H_cm: float, Df: float,
                 props: Properties, soil: SoilProfile,
                 footing_type: str = "Rectangular"):
        self.col1  = col1
        self.col2  = col2
        self.B     = B
        self.H_cm  = H_cm
        self.Df    = Df
        self.props = props
        self.soil  = soil
        self.ftype = footing_type
        self.d_cm  = H_cm - 7.5

    def _total_loads(self):
        P1 = self.col1.P_DL + self.col1.P_LL
        P2 = self.col2.P_DL + self.col2.P_LL
        P_total = P1 + P2
        # Resultant location from left edge
        x_R = (P1 * self.col1.x_pos + P2 * self.col2.x_pos) / P_total
        return P1, P2, P_total, x_R

    def design_rectangular(self):
        P1, P2, P_total, x_R = self._total_loads()
        # Length such that resultant is at centroid → L = 2*x_R (if col1 at 0)
        col1_edge = 0.3      # overhang from col1 face
        col2_edge = 0.3
        x1 = self.col1.x_pos
        x2 = self.col2.x_pos
        L  = 2 * (x_R - col1_edge) + col1_edge + col2_edge

        # Uniform bearing pressure (service)
        A  = self.B * L
        H_m = self.H_cm / 100
        W_ftg = A * H_m * 2.4
        W_soil= A * (self.Df - H_m) * self.soil.soil_density
        q_net  = (P_total) / A
        q_total = (P_total + W_ftg + W_soil) / A

        # Shear & moment diagrams (simplified beam model)
        n  = 200
        xs = np.linspace(0, L, n)
        qu_beam = (P_total * 1000) / (self.B * 100 * (L * 100))  # kg/cm²

        # Reactions are distributed (upward) – treat as beam on elastic foundation (simplified)
        w_up = P_total / L   # ton/m upward distributed

        # Shear diagram (integrating from left)
        V_arr = np.zeros(n)
        M_arr = np.zeros(n)
        for i, x in enumerate(xs):
            shear = -w_up * x
            if x >= x1:
                shear += P1
            if x >= x2:
                shear += P2
            V_arr[i] = shear

        for i, x in enumerate(xs):
            mom = -w_up * x ** 2 / 2
            if x >= x1:
                mom += P1 * (x - x1)
            if x >= x2:
                mom += P2 * (x - x2)
            M_arr[i] = mom

        M_max_pos = float(np.max(M_arr))
        M_max_neg = float(np.min(M_arr))

        return {
            "type": "Rectangular", "L": L, "B": self.B,
            "x_R": x_R, "P_total": P_total,
            "q_net": q_net, "q_total": q_total,
            "qu_beam": qu_beam,
            "xs": xs, "V_arr": V_arr, "M_arr": M_arr,
            "M_max_pos": M_max_pos, "M_max_neg": M_max_neg,
            "x1": x1, "x2": x2,
        }

    def design_trapezoidal(self):
        """
        Trapezoidal footing: adjust widths b1, b2 at each end so
        centroid of trapezoid coincides with resultant x_R.
        Centroid of trapezoid = L*(2*b2+b1)/(3*(b1+b2)) from b1 end.
        """
        P1, P2, P_total, x_R = self._total_loads()

        x1 = self.col1.x_pos
        x2 = self.col2.x_pos
        L  = x2 - x1 + 0.6   # total length with overhang

        # Centroid of trapezoid from left: xc = L*(2*b2+b1)/(3*(b1+b2)) = x_R
        # Let avg_b = (b1+b2)/2 = B (keep average width as input B)
        # → b1+b2 = 2B, and b1 & b2 from centroid equation:
        # xc*(b1+b2) = L*(2*b2+b1)/3 → 3*xc*2B = L*(2*b2+b1) → ...
        # Solve for b1,b2:
        # 3*xc*(b1+b2) = L*(b1 + 2*b2)
        # 3*xc*2B = L*b1 + 2*L*b2   and   b1 + b2 = 2B
        # → b1 = 2B*(3*xc/L - 1)*... (derived below)
        xc = x_R - (x1 - 0.3)   # centroid from left edge
        # System: b1+b2=2B, 3*xc*(b1+b2)=L*(b1+2*b2)
        # 6*B*xc = L*b1 + 2*L*b2 = L*(2B - b2) + 2*L*b2 = 2*B*L + L*b2
        b2 = (6 * self.B * xc - 2 * self.B * L) / L
        b1 = 2 * self.B - b2
        b1 = max(b1, 0.5); b2 = max(b2, 0.5)  # minimum 0.5 m

        A  = (b1 + b2) / 2 * L
        H_m = self.H_cm / 100
        W_ftg  = A * H_m * 2.4
        W_soil = A * (self.Df - H_m) * self.soil.soil_density
        q_net   = P_total / A
        q_total = (P_total + W_ftg + W_soil) / A

        n  = 200
        xs = np.linspace(0, L, n)
        # Width at each x (linear interpolation)
        bx = b1 + (b2 - b1) * xs / L
        # Upward pressure intensity (ton/m) varies with width
        q_up = P_total / A   # ton/m² (uniform net)
        w_up = q_up * bx      # ton/m at each x

        # Shear & moment (trapezoidal distributed upward load)
        V_arr = np.zeros(n)
        M_arr = np.zeros(n)
        dx = L / (n - 1)

        for i in range(n):
            x = xs[i]
            # cumulative upward force to left of x
            V_up = float(np.trapz(w_up[:i+1], xs[:i+1]))
            shear = -V_up
            if x >= (x1 - (x1 - 0.3)):
                shear += P1
            if x >= (x2 - (x1 - 0.3)):
                shear += P2
            V_arr[i] = shear

        for i in range(n):
            x = xs[i]
            M_up = float(np.trapz(
                [w_up[j] * (x - xs[j]) for j in range(i+1)],
                xs[:i+1]
            ))
            mom = -M_up
            dx_1 = x - (x1 - (x1 - 0.3))
            dx_2 = x - (x2 - (x1 - 0.3))
            if x >= (x1 - (x1 - 0.3)):
                mom += P1 * dx_1
            if x >= (x2 - (x1 - 0.3)):
                mom += P2 * dx_2
            M_arr[i] = mom

        return {
            "type": "Trapezoidal", "L": L, "b1": b1, "b2": b2,
            "x_R": x_R, "P_total": P_total,
            "q_net": q_net, "q_total": q_total,
            "xs": xs, "V_arr": V_arr, "M_arr": M_arr,
            "M_max_pos": float(np.max(M_arr)),
            "M_max_neg": float(np.min(M_arr)),
            "x1": x1, "x2": x2, "bx": bx,
        }


st.set_page_config(page_title="Advanced Biaxial Foundation Suite v3", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main-header { font-size:30px; font-weight:700; color:#0F172A; margin-bottom:5px; }
    .sub-header { font-size:15px; color:#475569; margin-bottom:25px; }
    .section-title { font-size:20px; font-weight:600; color:#1E3A8A; border-left:6px solid #2563EB; padding-left:12px; margin-top:25px; margin-bottom:15px; }
    .metric-card { background-color:#F8FAFC; padding:20px; border-radius:10px; border:1px solid #E2E8F0; }
    .status-pass { color:#059669; font-weight:700; }
    .status-fail { color:#DC2626; font-weight:700; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏗️ Advanced Biaxial Foundation Engineering Suite <span style="font-size:16px;color:#64748B;">v3</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ULS & SLS · ACI 318-19 · Terzaghi & Meyerhof Bearing Capacity · Elastic + Consolidation Settlement · Combined Footings · Rect. & Circular</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────
# SIDEBAR INPUTS
# ──────────────────────────────────────────
st.sidebar.header("📥 1. Structural Service Loads")
with st.sidebar.expander("📊 Axial, Moments & Shears", expanded=True):
    P_DL = st.number_input("Dead Load P_DL (tons)", min_value=0.0, value=30.0, step=1.0, key="PDL")
    P_LL = st.number_input("Live Load P_LL (tons)", min_value=0.0, value=18.0, step=1.0, key="PLL")
    st.sidebar.markdown("**Bending Moments**")
    M_DL_x = st.sidebar.number_input("M_DL x (ton-m)", value=3.5, step=0.5)
    M_LL_x = st.sidebar.number_input("M_LL x (ton-m)", value=2.0, step=0.5)
    M_WL_x = st.sidebar.number_input("M_WL x – Wind (ton-m)", value=1.5, step=0.5)
    M_DL_y = st.sidebar.number_input("M_DL y (ton-m)", value=2.5, step=0.5)
    M_LL_y = st.sidebar.number_input("M_LL y (ton-m)", value=1.5, step=0.5)
    M_WL_y = st.sidebar.number_input("M_WL y – Wind (ton-m)", value=1.0, step=0.5)
    st.sidebar.markdown("**Base Shears**")
    V_hx = st.sidebar.number_input("Horizontal Shear V_hx (tons)", value=2.0, step=0.1)
    V_hy = st.sidebar.number_input("Horizontal Shear V_hy (tons)", value=1.8, step=0.1)

st.sidebar.header("🧱 2. Material & Geotechnical Specs")
with st.sidebar.expander("Concrete & Steel", expanded=False):
    qa_allow   = st.sidebar.number_input("Allowable Bearing q_allow (ton/m²)", min_value=1.0, value=20.0, step=0.5)
    fc_prime   = st.sidebar.number_input("Concrete fc' (ksc)", min_value=150, value=280, step=10)
    fy         = st.sidebar.selectbox("Rebar fy", [3000, 4000], index=1,
                     format_func=lambda x: f"Grade 40 (fy={x} ksc)" if x == 3000 else f"SD40 (fy={x} ksc)")
    soil_density  = st.sidebar.number_input("Soil Density (ton/m³)", value=1.8, step=0.1)
    base_friction = st.sidebar.number_input("Base Friction μ", min_value=0.1, max_value=0.7, value=0.50, step=0.05)

with st.sidebar.expander("🌱 Soil Profile (Bearing & Settlement)", expanded=False):
    bc_method   = st.sidebar.selectbox("Bearing Capacity Method", ["Terzaghi", "Meyerhof", "Both"])
    fail_mode   = st.sidebar.selectbox("Terzaghi Failure Mode", ["General", "Local", "Punching"])
    cohesion    = st.sidebar.number_input("Cohesion c (ton/m²)", min_value=0.0, value=0.0, step=0.5)
    phi_deg     = st.sidebar.number_input("Friction Angle φ (°)", min_value=0.0, max_value=45.0, value=30.0, step=1.0)
    st.sidebar.markdown("**Elastic Settlement**")
    Es_soil     = st.sidebar.number_input("Elastic Modulus Es (ton/m²)", min_value=100.0, value=2000.0, step=100.0)
    nu_soil     = st.sidebar.number_input("Poisson's Ratio ν", min_value=0.1, max_value=0.49, value=0.30, step=0.01)
    st.sidebar.markdown("**Consolidation Settlement**")
    Cc_soil     = st.sidebar.number_input("Compression Index Cc", min_value=0.01, value=0.30, step=0.01)
    Cs_soil     = st.sidebar.number_input("Swelling Index Cs",    min_value=0.001, value=0.05, step=0.005)
    e0_soil     = st.sidebar.number_input("Initial Void Ratio e₀", min_value=0.1, value=0.80, step=0.05)
    OCR_soil    = st.sidebar.number_input("OCR", min_value=1.0, value=1.0, step=0.5)
    H_clay      = st.sidebar.number_input("Clay Layer Thickness (m)", min_value=0.5, value=3.0, step=0.5)
    sigma_v0    = st.sidebar.number_input("Initial Eff. Stress σ'v0 at mid-layer (ton/m²)", min_value=1.0, value=10.0, step=1.0)

st.sidebar.header("📐 3. Column Dimensions")
col_bx = st.sidebar.number_input("Column cx (cm)", value=40.0, step=5.0)
col_by = st.sidebar.number_input("Column cy (cm)", value=40.0, step=5.0)

# ──────────────────────────────────────────
# FOOTING GEOMETRY
# ──────────────────────────────────────────
st.markdown('<div class="section-title">📐 Footing Geometry</div>', unsafe_allow_html=True)

shape = st.radio("Footing Shape", ["Rectangular / Square", "Circular"], horizontal=True)
is_circular = shape == "Circular"

if is_circular:
    gc1, gc2, gc3 = st.columns(3)
    D_circ = gc1.number_input("Diameter D (m)", min_value=0.5, value=2.5, step=0.1)
    H_cm   = gc2.number_input("Thickness H (cm)", min_value=25.0, value=60.0, step=5.0)
    Df_m   = gc3.number_input("Embedment Df (m)", min_value=0.5, value=1.5, step=0.1)
    B_m = L_m = D_circ
else:
    gc1, gc2, gc3, gc4 = st.columns(4)
    B_m  = gc1.number_input("Width B (m)", min_value=1.0, value=2.5, step=0.1)
    L_m  = gc2.number_input("Length L (m)", min_value=1.0, value=2.5, step=0.1)
    H_cm = gc3.number_input("Thickness H (cm)", min_value=25.0, value=60.0, step=5.0)
    Df_m = gc4.number_input("Embedment Df (m)", min_value=0.5, value=1.5, step=0.1)
    D_circ = 0.0

# ──────────────────────────────────────────
# OPTIMIZATION PANEL
# ──────────────────────────────────────────
with st.expander("🔧 Auto-Optimization: Find Minimum Footing Size", expanded=False):
    opt_col1, opt_col2 = st.columns([2, 1])
    if is_circular:
        opt_note = opt_col1.info("Circular: will find minimum diameter D.")
        aspect = 1.0
    else:
        aspect = opt_col1.number_input("Aspect Ratio L/B", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    run_opt = opt_col2.button("🚀 Run Optimization", use_container_width=True)

    if run_opt:
        loads_opt = Loads(P_DL, P_LL, M_DL_x, M_LL_x, M_WL_x, M_DL_y, M_LL_y, M_WL_y, V_hx, V_hy)
        props_opt = Properties(qa_allow, fc_prime, fy, soil_density, base_friction)
        geo_opt   = Geometry(B_m, L_m, H_cm, Df_m, col_bx, col_by, "circular" if is_circular else "rectangular", D_circ)
        des_opt   = FoundationDesigner(loads_opt, props_opt, geo_opt)
        with st.spinner("Optimizing..."):
            B_opt, L_opt = des_opt.optimize_dimensions(aspect_ratio=aspect)
        if is_circular:
            st.success(f"✅ Minimum diameter: **D = {B_opt:.1f} m**  (H={H_cm:.0f} cm kept fixed)")
        else:
            st.success(f"✅ Minimum size: **B = {B_opt:.1f} m × L = {L_opt:.1f} m**  (H={H_cm:.0f} cm kept fixed)")
        st.caption("Tip: paste these values back into the geometry inputs above to verify.")

# ──────────────────────────────────────────
# RUN ENGINE
# ──────────────────────────────────────────
loads    = Loads(P_DL, P_LL, M_DL_x, M_LL_x, M_WL_x, M_DL_y, M_LL_y, M_WL_y, V_hx, V_hy)
props    = Properties(qa_allow, fc_prime, fy, soil_density, base_friction)
geo      = Geometry(B_m, L_m, H_cm, Df_m, col_bx, col_by,
                    "circular" if is_circular else "rectangular", D_circ)
soil     = SoilProfile(
    cohesion=cohesion, phi_deg=phi_deg,
    Es=Es_soil, nu=nu_soil,
    Cc=Cc_soil, Cs=Cs_soil, e0=e0_soil, OCR=OCR_soil,
    H_clay=H_clay, sigma_v0=sigma_v0,
    soil_density=soil_density,
    method="Terzaghi" if bc_method == "Terzaghi" else "Meyerhof",
    failure_mode=fail_mode,
)

designer = FoundationDesigner(loads, props, geo)
sls      = designer.analyze_service_state()
uls      = designer.analyze_ultimate_state()
bars_x, space_x = designer.design_flexure(uls["M_ux"], designer.L_cm)
bars_y, space_y = designer.design_flexure(uls["M_uy"], designer.B_cm)

# Bearing capacity
bc_engine = BearingCapacityEngine(soil, geo, loads)
if bc_method == "Both":
    bc_results = bc_engine.run_both()
elif bc_method == "Terzaghi":
    bc_results = {"Terzaghi": bc_engine.terzaghi()}
else:
    bc_results = {"Meyerhof": bc_engine.meyerhof()}

# Settlement
q_net_service = sls["P_total"] / designer.A_base - soil_density * Df_m
sett_engine   = SettlementEngine(soil, geo, max(q_net_service, 0))
sett_results  = sett_engine.total()

# ──────────────────────────────────────────
# HELPER: utilization bar
# ──────────────────────────────────────────
def render_bar(label, demand, capacity, is_fs=False):
    if is_fs:
        ratio = capacity / demand if demand > 0 else 0.5
        val_str, cap_str = f"{demand:.2f}", f"≥ {capacity:.2f}"
    else:
        ratio = demand / capacity if capacity > 0 else 1.5
        val_str, cap_str = f"{demand:.2f}", f"≤ {capacity:.2f}"
    pct = min(ratio * 100, 100)
    color = "#10B981" if ratio <= 0.80 else ("#F59E0B" if ratio <= 1.0 else "#EF4444")
    status = "PASS" if ratio <= 1.0 else "FAIL"
    return f"""
    <div style="background:#F8FAFC;padding:14px;border-radius:8px;border:1px solid #E2E8F0;margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;font-weight:600;color:#1E293B;margin-bottom:7px;">
        <span style="font-size:14px;">{label}</span>
        <span style="color:{color};font-size:13px;">Demand: {val_str} | Limit: {cap_str}
          <span style="background:{color};color:white;padding:2px 7px;border-radius:4px;margin-left:8px;font-size:11px;">{status}</span>
        </span>
      </div>
      <div style="width:100%;background:#CBD5E1;border-radius:5px;height:10px;overflow:hidden;">
        <div style="width:{pct}%;background:{color};height:100%;"></div>
      </div>
      <div style="text-align:right;margin-top:4px;font-size:11px;color:#64748B;">Utilization: {ratio*100:.1f}%</div>
    </div>"""

# ──────────────────────────────────────────
# TABS
# ──────────────────────────────────────────
tab_dash, tab_geo, tab_bearing, tab_settle, tab_combined, tab_struct, tab_3d, tab_draw = st.tabs([
    "📊 Safety Dashboard",
    "🪨 Geotechnical",
    "🏔️ Bearing Capacity",
    "📉 Settlement",
    "🔗 Combined Footing",
    "🧱 Structural Design",
    "🌐 3D Soil Pressure",
    "🎨 Blueprints",
])

# ════════════════════════════════════════════
# TAB 1: DASHBOARD
# ════════════════════════════════════════════
with tab_dash:
    st.subheader("💡 Foundation Performance Dashboard")

    # Governing load combo banner
    combo_color = "#1D4ED8" if "W" in uls["governing_combo"] else "#059669"
    st.markdown(
        f'<div style="background:{combo_color}10;border:1px solid {combo_color};border-radius:8px;'
        f'padding:10px 16px;margin-bottom:16px;color:{combo_color};font-weight:600;">'
        f'⚡ Governing ULS Load Combination: <code>{uls["governing_combo"]}</code>'
        f'  |  P_u = {uls["P_u"]:.1f} t  |  M_ux = {uls["M_u_x"]:.2f} t·m  |  M_uy = {uls["M_u_y"]:.2f} t·m'
        f'</div>',
        unsafe_allow_html=True
    )

    col_geo_d, col_str_d = st.columns(2)
    with col_geo_d:
        st.markdown("### 🪨 Geotechnical Safety")
        st.markdown(render_bar("Max Soil Bearing (t/m²)", sls["q_max"], qa_allow), unsafe_allow_html=True)
        st.markdown(render_bar("FS Overturning – X axis", sls["FS_ovr_x"], 1.50, is_fs=True), unsafe_allow_html=True)
        st.markdown(render_bar("FS Overturning – Y axis", sls["FS_ovr_y"], 1.50, is_fs=True), unsafe_allow_html=True)
        st.markdown(render_bar("FS Sliding",              sls["FS_slide"],  1.50, is_fs=True), unsafe_allow_html=True)

    with col_str_d:
        st.markdown("### 🧱 Structural Safety")
        st.markdown(render_bar("One-Way Shear X (kg/cm²)", uls["v_u_wide_x"],   uls["phi_v_c_wide"]),  unsafe_allow_html=True)
        st.markdown(render_bar("One-Way Shear Y (kg/cm²)", uls["v_u_wide_y"],   uls["phi_v_c_wide"]),  unsafe_allow_html=True)
        st.markdown(render_bar("Punching Shear (kg/cm²)",  uls["v_u_punch"],    uls["phi_v_c_punch"]), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Construction Specification Summary")
    dim_label = f"D = {D_circ:.2f} m (circular)" if is_circular else f"B = {B_m:.2f} m × L = {L_m:.2f} m"
    st.info(f"📐 **Footing:** {dim_label} | H = **{H_cm:.0f} cm** | Effective depth d = **{designer.d_cm:.1f} cm**")
    c1, c2 = st.columns(2)
    c1.success(f"🧲 X-direction (along B): **{bars_x}×DB16** @ {space_x:.1f} cm c/c")
    c2.success(f"🧲 Y-direction (along L): **{bars_y}×DB16** @ {space_y:.1f} cm c/c")

    # Detailed calcs expanders
    with st.expander("🔍 Detailed Geotechnical Calculations"):
        st.markdown("### 1. Soil Bearing Pressure")
        st.latex(r"q_{max} = \frac{P_{total}}{A} + \frac{M_x}{W_x} + \frac{M_y}{W_y}")
        geo1 = "✅ PASS" if sls["q_max"] <= qa_allow else "❌ FAIL"
        st.markdown(f"$q_{{max}} = {sls['q_max']:.2f}$ t/m² vs $q_{{allow}} = {qa_allow:.2f}$ t/m²  →  **{geo1}**")

        st.markdown("---")
        st.markdown("### 2. Overturning – X axis")
        st.latex(r"FS_{ovr,x} = \frac{P_{total} \cdot L/2}{M_x + V_{hy} \cdot D_f}")
        geo2 = "✅ PASS" if sls["FS_ovr_x"] >= 1.5 else "❌ FAIL"
        st.markdown(f"$FS_{{ovr,x}} = {sls['FS_ovr_x']:.2f}$ ≥ 1.50  →  **{geo2}**")

        st.markdown("---")
        st.markdown("### 3. Overturning – Y axis")
        st.latex(r"FS_{ovr,y} = \frac{P_{total} \cdot B/2}{M_y + V_{hx} \cdot D_f}")
        geo3 = "✅ PASS" if sls["FS_ovr_y"] >= 1.5 else "❌ FAIL"
        st.markdown(f"$FS_{{ovr,y}} = {sls['FS_ovr_y']:.2f}$ ≥ 1.50  →  **{geo3}**")

        st.markdown("---")
        st.markdown("### 4. Sliding")
        st.latex(r"FS_{slide} = \frac{P_{total} \cdot \mu}{V_h}")
        geo4 = "✅ PASS" if sls["FS_slide"] >= 1.5 else "❌ FAIL"
        st.markdown(f"$FS_{{slide}} = {sls['FS_slide']:.2f}$ ≥ 1.50  →  **{geo4}**")

    with st.expander("🔍 Detailed Structural Calculations"):
        st.markdown(f"**Governing combo:** `{uls['governing_combo']}`")
        st.markdown("### 1. One-Way Shear – X direction")
        st.latex(r"v_{u,x} = \frac{q_{u,x} \cdot L \cdot c_{x,crit}}{L \cdot d}")
        s1 = "✅ PASS" if uls["v_u_wide_x"] <= uls["phi_v_c_wide"] else "❌ FAIL"
        st.markdown(f"$v_{{u,x}} = {uls['v_u_wide_x']:.3f}$ kg/cm² vs $\\phi v_c = {uls['phi_v_c_wide']:.3f}$ kg/cm²  →  **{s1}**")

        st.markdown("### 2. One-Way Shear – Y direction")
        s2 = "✅ PASS" if uls["v_u_wide_y"] <= uls["phi_v_c_wide"] else "❌ FAIL"
        st.markdown(f"$v_{{u,y}} = {uls['v_u_wide_y']:.3f}$ kg/cm² vs $\\phi v_c = {uls['phi_v_c_wide']:.3f}$ kg/cm²  →  **{s2}**")

        st.markdown("### 3. Punching Shear")
        st.latex(r"v_{u,punch} = \frac{q_{u,max} \cdot (A - A_{punch})}{b_0 \cdot d}")
        s3 = "✅ PASS" if uls["v_u_punch"] <= uls["phi_v_c_punch"] else "❌ FAIL"
        st.latex(rf"b_0 = {uls['bo']:.1f}\text{{ cm}},\quad v_{{u}} = {uls['v_u_punch']:.3f}\text{{ kg/cm}}^2 \le \phi v_c = {uls['phi_v_c_punch']:.3f}\text{{ kg/cm}}^2 \rightarrow \textbf{{{s3}}}")

        st.markdown("### 4. Development Length")
        st.latex(r"L_d = \frac{f_y}{1.4\sqrt{f'_c}} \cdot \frac{d_b}{10}")
        s4x = "✅ PASS" if uls["available_L_d_x"] >= uls["L_d"] else "⚠️ HOOK REQUIRED"
        s4y = "✅ PASS" if uls["available_L_d_y"] >= uls["L_d"] else "⚠️ HOOK REQUIRED"
        st.markdown(f"$L_d = {uls['L_d']:.1f}$ cm  |  Available X: {uls['available_L_d_x']:.1f} cm → **{s4x}** | Available Y: {uls['available_L_d_y']:.1f} cm → **{s4y}**")

# ════════════════════════════════════════════
# TAB 2: GEOTECHNICAL
# ════════════════════════════════════════════
with tab_geo:
    st.markdown("### 🪨 Geotechnical Analytics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Vertical Load P_total", f"{sls['P_total']:.2f} t")
        st.metric("Eccentricity e_x", f"{sls['e_x']:.4f} m", delta=f"kern = {sls['kern_x']:.3f} m")
        st.metric("Eccentricity e_y", f"{sls['e_y']:.4f} m", delta=f"kern = {sls['kern_y']:.3f} m")
    with col_b:
        st.metric("q_max", f"{sls['q_max']:.2f} t/m²", delta=f"Limit {qa_allow} t/m²")
        st.metric("q_min", f"{sls['q_min']:.2f} t/m²")
        tension_str = "⚠️ YES – partial liftoff" if sls["has_tension"] else "✅ NO – full contact"
        st.metric("Tension Zone?", tension_str)
    st.markdown("---")
    st.write("#### Global Safety Factors")
    cols = st.columns(3)
    cols[0].metric("FS Overturning X", f"{sls['FS_ovr_x']:.2f}", delta="Req ≥ 1.50")
    cols[1].metric("FS Overturning Y", f"{sls['FS_ovr_y']:.2f}", delta="Req ≥ 1.50")
    cols[2].metric("FS Sliding",       f"{sls['FS_slide']:.2f}",  delta="Req ≥ 1.50")

# ════════════════════════════════════════════
# TAB 3: BEARING CAPACITY
# ════════════════════════════════════════════
with tab_bearing:
    st.markdown("### 🏔️ Ultimate Bearing Capacity Analysis")

    def _bc_table(res, label):
        qu   = res["qu_ultimate"]
        qa   = res["qa_computed"]
        pass_ = qa >= sls["q_max"]
        color = "#059669" if pass_ else "#DC2626"
        status = "✅ PASS" if pass_ else "❌ FAIL"

        st.markdown(f"#### {label}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("qᵤ ultimate (t/m²)",    f"{qu:.2f}")
        c2.metric("qₐ computed (t/m²)",    f"{qa:.2f}")
        c3.metric("FS bearing",             f"{res['FS_bearing']:.1f}")
        c4.metric("q_max actual (t/m²)",   f"{sls['q_max']:.2f}",
                  delta=f"{status}")

        st.markdown(f"""
        <div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px;margin-top:8px;'>
        <b>Bearing Capacity Factors:</b>&nbsp;
        N<sub>c</sub> = {res['Nc']:.2f} &nbsp;|&nbsp;
        N<sub>q</sub> = {res['Nq']:.2f} &nbsp;|&nbsp;
        N<sub>γ</sub> = {res['Ng']:.2f}
        </div>""", unsafe_allow_html=True)

        if "sc" in res:   # Meyerhof extras
            st.markdown(f"""
            <div style='background:#F0F9FF;border:1px solid #BAE6FD;border-radius:8px;padding:10px;margin-top:6px;font-size:13px;'>
            <b>Shape:</b> s<sub>c</sub>={res['sc']:.3f}, s<sub>q</sub>={res['sq']:.3f}, s<sub>γ</sub>={res['sg']:.3f} &nbsp;|&nbsp;
            <b>Depth:</b> d<sub>c</sub>={res['dc']:.3f}, d<sub>q</sub>={res['dq']:.3f}, d<sub>γ</sub>={res['dg']:.3f} &nbsp;|&nbsp;
            <b>Inclin:</b> i<sub>c</sub>={res['ic']:.3f}, i<sub>q</sub>={res['iq']:.3f}, i<sub>γ</sub>={res['ig']:.3f}
            (α={res['alpha_deg']:.1f}°)
            </div>""", unsafe_allow_html=True)

        with st.expander(f"📐 Detailed Formula – {label}"):
            if "Terzaghi" in label:
                st.latex(r"q_u = c \cdot N_c \cdot s_c + \gamma D_f \cdot N_q \cdot s_q + 0.5\,\gamma\,B\,N_\gamma \cdot s_\gamma")
                st.markdown(f"**c** = {res.get('c_use', cohesion):.2f} t/m², φ = {res.get('phi_use_deg', phi_deg):.1f}°")
            else:
                st.latex(r"q_u = c\,N_c\,s_c\,d_c\,i_c + \gamma D_f N_q s_q d_q i_q + 0.5\,\gamma\,B\,N_\gamma\,s_\gamma\,d_\gamma\,i_\gamma")
            st.latex(rf"q_u = {qu:.2f}\;\text{{t/m}}^2 \quad\Rightarrow\quad q_a = \frac{{q_u}}{{FS}} = \frac{{{qu:.2f}}}{{3.0}} = {qa:.2f}\;\text{{t/m}}^2")
            st.markdown(f"**Verdict:** q_a ({qa:.2f}) {'≥' if pass_ else '<'} q_max ({sls['q_max']:.2f}) → **{status}**")

    for label, res in bc_results.items():
        _bc_table(res, label)
        st.markdown("---")

    # φ sensitivity chart
    st.markdown("#### 📈 qu vs Friction Angle φ (sensitivity)")
    phi_range = list(range(0, 46, 2))
    import copy as _copy
    qu_terz, qu_meyh = [], []
    for p in phi_range:
        s2 = _copy.copy(soil); s2.phi_deg = p
        eng = BearingCapacityEngine(s2, geo, loads)
        qu_terz.append(eng.terzaghi()["qu_ultimate"])
        qu_meyh.append(eng.meyerhof()["qu_ultimate"])

    fig_bc = go.Figure()
    fig_bc.add_trace(go.Scatter(x=phi_range, y=qu_terz, name="Terzaghi", line=dict(color="#2563EB", width=2)))
    fig_bc.add_trace(go.Scatter(x=phi_range, y=qu_meyh, name="Meyerhof", line=dict(color="#059669", width=2)))
    fig_bc.add_hline(y=sls["q_max"]*3, line_dash="dash", line_color="#DC2626",
                     annotation_text=f"3×q_max = {sls['q_max']*3:.1f}")
    fig_bc.update_layout(xaxis_title="φ (°)", yaxis_title="qᵤ (t/m²)",
                         height=350, margin=dict(t=20, b=40))
    st.plotly_chart(fig_bc, use_container_width=True)


# ════════════════════════════════════════════
# TAB 4: SETTLEMENT
# ════════════════════════════════════════════
with tab_settle:
    st.markdown("### 📉 Settlement Analysis")
    el  = sett_results["elastic"]
    con = sett_results["consolidation"]
    St  = sett_results["St_cm"]

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Elastic Settlement Sᵢ",        f"{el['Si_cm']:.1f} cm")
    m2.metric("Consolidation Settlement Sc",   f"{con['Sc_cm']:.1f} cm")
    m3.metric("Total Settlement Sₜ",           f"{St:.1f} cm",
              delta="⚠️ Check if > 2.5 cm" if St > 2.5 else "✅ Acceptable")

    st.markdown("---")
    col_el, col_con = st.columns(2)

    with col_el:
        st.markdown("#### Elastic Settlement")
        st.latex(r"S_i = q_{net} \cdot B \cdot \frac{1-\nu^2}{E_s} \cdot I_f")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | q_net | {q_net_service:.2f} t/m² |
        | B | {B_m:.2f} m |
        | Es | {Es_soil:.0f} t/m² |
        | ν | {nu_soil:.2f} |
        | Influence factor If | {el['If']:.3f} |
        | **Sᵢ** | **{el['Si_cm']:.2f} cm** |
        """)

    with col_con:
        st.markdown("#### Consolidation Settlement")
        st.latex(r"S_c = \frac{C_c \cdot H}{1+e_0} \log_{10}\frac{\sigma'_{v0}+\Delta\sigma}{\sigma'_{v0}}")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Regime | {con['regime']} |
        | σ'v0 | {con['sv0']:.2f} t/m² |
        | σ'vc (precon.) | {con['svc']:.2f} t/m² |
        | Δσ (Boussinesq) | {con['dsigma']:.2f} t/m² |
        | σ'v1 (final) | {con['sv1']:.2f} t/m² |
        | Cc | {Cc_soil:.3f} |
        | e₀ | {e0_soil:.2f} |
        | H_clay | {H_clay:.1f} m |
        | **Sc** | **{con['Sc_cm']:.2f} cm** |
        """)

    # Settlement vs B chart
    st.markdown("---")
    st.markdown("#### 📊 Total Settlement vs Footing Width B")
    B_range = np.linspace(1.0, 6.0, 30)
    St_arr  = []
    for Bv in B_range:
        g2 = _copy.copy(geo); g2.B = Bv; g2.L = Bv
        q2 = max(sls["P_total"] / (Bv * Bv) - soil_density * Df_m, 0)
        se = SettlementEngine(soil, g2, q2)
        St_arr.append(se.total()["St_cm"])

    fig_st = go.Figure()
    fig_st.add_trace(go.Scatter(x=list(B_range), y=St_arr,
                                line=dict(color="#7C3AED", width=2), name="Total Settlement"))
    fig_st.add_hline(y=2.5, line_dash="dash", line_color="#DC2626",
                     annotation_text="Typical limit 2.5 cm")
    fig_st.add_vline(x=B_m, line_dash="dot", line_color="#2563EB",
                     annotation_text=f"Current B={B_m}m")
    fig_st.update_layout(xaxis_title="B (m)", yaxis_title="Settlement (cm)",
                         height=350, margin=dict(t=20, b=40))
    st.plotly_chart(fig_st, use_container_width=True)


# ════════════════════════════════════════════
# TAB 5: COMBINED FOOTING
# ════════════════════════════════════════════
with tab_combined:
    st.markdown("### 🔗 Combined Footing Design (2 Columns)")

    cf1, cf2 = st.columns(2)
    with cf1:
        st.markdown("**Column 1**")
        c1_PDL = st.number_input("P_DL col1 (ton)", value=25.0, step=1.0, key="c1pdl")
        c1_PLL = st.number_input("P_LL col1 (ton)", value=15.0, step=1.0, key="c1pll")
        c1_cx  = st.number_input("cx col1 (cm)", value=40.0, step=5.0, key="c1cx")
        c1_cy  = st.number_input("cy col1 (cm)", value=40.0, step=5.0, key="c1cy")
        c1_x   = st.number_input("Position x1 from left edge (m)", value=0.30, step=0.1, key="c1x")

    with cf2:
        st.markdown("**Column 2**")
        c2_PDL = st.number_input("P_DL col2 (ton)", value=35.0, step=1.0, key="c2pdl")
        c2_PLL = st.number_input("P_LL col2 (ton)", value=20.0, step=1.0, key="c2pll")
        c2_cx  = st.number_input("cx col2 (cm)", value=40.0, step=5.0, key="c2cx")
        c2_cy  = st.number_input("cy col2 (cm)", value=40.0, step=5.0, key="c2cy")
        c2_x   = st.number_input("Position x2 from left edge (m)", value=3.70, step=0.1, key="c2x")

    cf_B   = st.number_input("Footing Width B (m)", min_value=0.5, value=2.0, step=0.1, key="cfB")
    cf_H   = st.number_input("Footing Thickness H (cm)", min_value=30.0, value=70.0, step=5.0, key="cfH")
    cf_type = st.radio("Footing Type", ["Rectangular", "Trapezoidal"], horizontal=True, key="cftype")

    col1d = ColumnData(c1_PDL, c1_PLL, c1_cx, c1_cy, c1_x)
    col2d = ColumnData(c2_PDL, c2_PLL, c2_cx, c2_cy, c2_x)
    cf_des = CombinedFootingDesigner(col1d, col2d, cf_B, cf_H, Df_m, props, soil, cf_type)

    try:
        if cf_type == "Rectangular":
            cfr = cf_des.design_rectangular()
        else:
            cfr = cf_des.design_trapezoidal()

        # Summary
        st.markdown("---")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Load",      f"{cfr['P_total']:.1f} t")
        mc2.metric("Footing Length L", f"{cfr['L']:.2f} m")
        mc3.metric("Net q (service)", f"{cfr['q_net']:.2f} t/m²")
        ok_q = cfr["q_net"] <= qa_allow
        mc4.metric("vs q_allow",      f"{qa_allow:.1f} t/m²",
                   delta="✅ OK" if ok_q else "❌ Exceeds limit")

        if cf_type == "Trapezoidal":
            t1, t2 = st.columns(2)
            t1.metric("Width at Col1 end b₁", f"{cfr['b1']:.2f} m")
            t2.metric("Width at Col2 end b₂", f"{cfr['b2']:.2f} m")

        # Shear & moment diagrams
        st.markdown("#### Shear Force & Bending Moment Diagrams")
        xs = cfr["xs"]; V = cfr["V_arr"]; M = cfr["M_arr"]

        fig_vm = go.Figure()
        fig_vm.add_trace(go.Scatter(x=xs, y=V, name="Shear V (ton)",
                                    line=dict(color="#2563EB", width=2), fill="tozeroy",
                                    fillcolor="rgba(37,99,235,0.1)"))
        fig_vm.add_trace(go.Scatter(x=xs, y=M, name="Moment M (ton·m)",
                                    line=dict(color="#DC2626", width=2), fill="tozeroy",
                                    fillcolor="rgba(220,38,38,0.1)"))
        fig_vm.add_vline(x=cfr["x1"], line_dash="dot", line_color="#059669",
                         annotation_text="Col1")
        fig_vm.add_vline(x=cfr["x2"], line_dash="dot", line_color="#059669",
                         annotation_text="Col2")
        fig_vm.update_layout(xaxis_title="x (m)", yaxis_title="V (ton) / M (ton·m)",
                             height=380, legend=dict(x=0.01, y=0.99),
                             margin=dict(t=20, b=40))
        st.plotly_chart(fig_vm, use_container_width=True)

        st.markdown(f"""
        | | Value |
        |---|---|
        | Max Positive Moment | {cfr['M_max_pos']:.2f} ton·m |
        | Max Negative Moment | {cfr['M_max_neg']:.2f} ton·m |
        """)

        # Plan sketch
        st.markdown("#### Plan Sketch")
        fig_cf, ax_cf = plt.subplots(figsize=(12, 4))
        fig_cf.patch.set_facecolor("#FFFFFF")
        L_cf = cfr["L"]

        if cf_type == "Trapezoidal":
            b1 = cfr["b1"]; b2 = cfr["b2"]
            poly_x = [0,     L_cf,  L_cf,     0,     0]
            poly_y = [-b1/2, -b2/2, b2/2,  b1/2, -b1/2]
            ax_cf.fill(poly_x, poly_y, facecolor="#F1F5F9", edgecolor="#0F172A", lw=2)
        else:
            ax_cf.add_patch(plt.Rectangle((0, -cf_B/2), L_cf, cf_B,
                                          facecolor="#F1F5F9", edgecolor="#0F172A", lw=2))

        for col_x_pos, cxs, cys, label in [
            (cfr["x1"], c1_cx/100, c1_cy/100, "C1"),
            (cfr["x2"], c2_cx/100, c2_cy/100, "C2"),
        ]:
            ax_cf.add_patch(plt.Rectangle(
                (col_x_pos - cxs/2, -cys/2), cxs, cys,
                facecolor="#FEE2E2", hatch="//", edgecolor="#DC2626", lw=1.5))
            ax_cf.text(col_x_pos, 0, label, ha="center", va="center",
                       fontsize=10, fontweight="bold", color="#DC2626")

        ax_cf.set_xlim(-0.3, L_cf + 0.3)
        ax_cf.set_ylim(-max(cf_B, 2)/2 - 0.4, max(cf_B, 2)/2 + 0.4)
        ax_cf.set_aspect("equal")
        ax_cf.set_title(f"{cf_type} Combined Footing — Plan View", fontsize=11,
                        fontweight="bold", color="#1E3A8A")
        ax_cf.axis("off")
        st.pyplot(fig_cf)

    except Exception as e:
        st.error(f"Combined footing calculation error: {e}")


with tab_struct:
    st.markdown("### 🧱 ULS Concrete Verification")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("One-Way Shear")
        st.metric("v_u,x", f"{uls['v_u_wide_x']:.3f} kg/cm²")
        st.metric("v_u,y", f"{uls['v_u_wide_y']:.3f} kg/cm²")
        st.metric("φv_c",  f"{uls['phi_v_c_wide']:.3f} kg/cm²")
        ok = uls['v_u_wide_max'] <= uls['phi_v_c_wide']
        st.success("✔️ PASS") if ok else st.error("❌ FAIL – Increase H")
    with col_s2:
        st.subheader("Punching Shear")
        st.metric("b₀",    f"{uls['bo']:.1f} cm")
        st.metric("v_u",   f"{uls['v_u_punch']:.3f} kg/cm²")
        st.metric("φv_c",  f"{uls['phi_v_c_punch']:.3f} kg/cm²")
        ok2 = uls['v_u_punch'] <= uls['phi_v_c_punch']
        st.success("✔️ PASS") if ok2 else st.error("❌ FAIL – Increase H")

    st.markdown("---")
    st.subheader("Flexural Reinforcement Summary")
    r1, r2 = st.columns(2)
    r1.info(f"**X-direction (bottom bars along B)**\n\n{bars_x} × DB16 @ {space_x:.1f} cm c/c")
    r2.info(f"**Y-direction (bottom bars along L)**\n\n{bars_y} × DB16 @ {space_y:.1f} cm c/c")

    st.subheader("Development Length")
    dl1, dl2, dl3 = st.columns(3)
    dl1.metric("Required L_d", f"{uls['L_d']:.1f} cm")
    dl2.metric("Available (X)", f"{uls['available_L_d_x']:.1f} cm",
               delta="OK" if uls['available_L_d_x'] >= uls['L_d'] else "⚠️ Hook needed")
    dl3.metric("Available (Y)", f"{uls['available_L_d_y']:.1f} cm",
               delta="OK" if uls['available_L_d_y'] >= uls['L_d'] else "⚠️ Hook needed")

# ════════════════════════════════════════════
# TAB 4: 3D SOIL PRESSURE SURFACE
# ════════════════════════════════════════════
with tab_3d:
    st.markdown("### 🌐 3D Contact Stress Distribution")
    st.caption("Service state pressures — shows biaxial gradient across footing base.")

    XX, YY, ZZ = designer.soil_pressure_grid(n=50)

    fig3d = go.Figure(data=[
        go.Surface(
            x=XX, y=YY, z=ZZ,
            colorscale="RdYlGn_r",
            cmin=0,
            cmax=float(np.nanmax(ZZ)) * 1.05,
            colorbar=dict(title="q (t/m²)", thickness=15),
            opacity=0.92,
        )
    ])
    # Allowable pressure plane
    fig3d.add_surface(
        x=XX, y=YY,
        z=np.full_like(ZZ, qa_allow),
        colorscale=[[0, "rgba(239,68,68,0.25)"], [1, "rgba(239,68,68,0.25)"]],
        showscale=False, name="q_allow",
        opacity=0.35,
    )
    fig3d.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Pressure (t/m²)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Contact Stress Distribution (red plane = q_allow)", x=0.5),
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.info(f"q_max = **{sls['q_max']:.2f}** t/m²  |  q_min = **{sls['q_min']:.2f}** t/m²  |  q_allow = **{qa_allow:.2f}** t/m²")

# ════════════════════════════════════════════
# TAB 5: BLUEPRINTS
# ════════════════════════════════════════════
with tab_draw:
    st.markdown("### 🎨 2D Engineering Blueprints")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.2]})
    fig.patch.set_facecolor("#FFFFFF")

    def draw_dim(ax, x1, y1, x2, y2, text, off_x=0, off_y=0):
        ax.plot([x1, x2], [y1, y2], color="#475569", lw=1)
        ax.plot([x1, x1], [y1 - 0.05, y1 + 0.05], color="#475569", lw=1.5)
        ax.plot([x2, x2], [y2 - 0.05, y2 + 0.05], color="#475569", lw=1.5)
        ax.text((x1+x2)/2+off_x, (y1+y2)/2+off_y, text,
                ha="center", va="center", color="#0F172A", fontsize=9, backgroundcolor="white")

    # ── PLAN VIEW ────────────────────────────────────────────────
    if is_circular:
        R = D_circ / 2
        circle = plt.Circle((R, R), R, facecolor="#F8FAFC", edgecolor="#0F172A", lw=2)
        ax1.add_patch(circle)
        ax1.plot([0.2, 2*R - 0.2], [R, R], color="#94A3B8", lw=1, linestyle="dashdot")
        ax1.plot([R, R], [0.2, 2*R - 0.2], color="#94A3B8", lw=1, linestyle="dashdot")
        col_x0 = R - (col_bx/100)/2
        col_y0 = R - (col_by/100)/2
        ax1.add_patch(plt.Rectangle((col_x0, col_y0), col_bx/100, col_by/100,
                                    facecolor="#FEE2E2", hatch="//", edgecolor="#DC2626", lw=1.5))
        p_off = (designer.d_cm/100)/2
        punch_circ = plt.Circle((R, R),
                                math.sqrt(((col_bx/100)/2+p_off)**2+((col_by/100)/2+p_off)**2),
                                fill=False, edgecolor="#D97706", lw=1.5, linestyle="--")
        ax1.add_patch(punch_circ)
        ax1.set_xlim(-0.3, 2*R + 0.3)
        ax1.set_ylim(-0.3, 2*R + 0.5)
        draw_dim(ax1, 0, 2*R+0.2, 2*R, 2*R+0.2, f"D = {D_circ:.2f} m", off_y=0.05)
    else:
        ax1.add_patch(plt.Rectangle((0, 0), B_m, L_m,
                                    facecolor="#F8FAFC", edgecolor="#0F172A", lw=2))
        ax1.plot([-0.2, B_m+0.2], [L_m/2, L_m/2], color="#94A3B8", lw=1, linestyle="dashdot")
        ax1.plot([B_m/2, B_m/2], [-0.2, L_m+0.2], color="#94A3B8", lw=1, linestyle="dashdot")
        col_x0 = (B_m - col_bx/100)/2
        col_y0 = (L_m - col_by/100)/2
        ax1.add_patch(plt.Rectangle((col_x0, col_y0), col_bx/100, col_by/100,
                                    facecolor="#FEE2E2", hatch="//", edgecolor="#DC2626", lw=1.5))
        p_off = (designer.d_cm/100)/2
        ax1.add_patch(plt.Rectangle((col_x0-p_off, col_y0-p_off),
                                    col_bx/100+2*p_off, col_by/100+2*p_off,
                                    fill=False, edgecolor="#D97706", lw=1.5, linestyle="--"))
        cover = 0.075
        for i in range(1, 4):
            ax1.plot([cover, B_m-cover], [cover+i*(L_m-2*cover)/15]*2, color="#3B82F6", lw=1, alpha=0.6)
            ax1.plot([cover+i*(B_m-2*cover)/15]*2, [cover, L_m-cover], color="#3B82F6", lw=1, alpha=0.6)
        draw_dim(ax1, 0, L_m+0.15, B_m, L_m+0.15, f"B = {B_m:.2f} m", off_y=0.05)
        ax1.plot([-0.15]*2, [0, L_m], color="#475569", lw=1)
        ax1.plot([-0.2, -0.1], [0, 0], color="#475569", lw=1.5)
        ax1.plot([-0.2, -0.1], [L_m, L_m], color="#475569", lw=1.5)
        ax1.text(-0.25, L_m/2, f"L = {L_m:.2f} m", ha="center", va="center",
                 color="#0F172A", fontsize=9, rotation=90)
        ax1.set_xlim(-0.4, B_m+0.3)
        ax1.set_ylim(-0.4, L_m+0.4)

    ax1.set_aspect("equal")
    ax1.set_title("PLAN VIEW", fontsize=12, fontweight="bold", color="#1E3A8A", pad=15)
    ax1.axis("off")

    # ── SECTION VIEW ─────────────────────────────────────────────
    lean_thick   = 0.05
    fbase        = lean_thick
    H_m          = H_cm / 100
    cover_m      = 0.075

    ax2.plot([-0.5, B_m+0.5], [Df_m, Df_m], color="#451A03", lw=1.5)
    ax2.text(B_m+0.1, Df_m+0.05, "F.G.L.", color="#451A03", fontsize=9, fontweight="bold")
    ax2.add_patch(plt.Rectangle((-0.05, 0), B_m+0.1, lean_thick,
                                facecolor="#E2E8F0", hatch="...", edgecolor="#64748B", lw=1))
    ax2.text(B_m+0.15, lean_thick/2, "Lean 5cm", color="#64748B", fontsize=8, va="center")
    ax2.add_patch(plt.Rectangle((0, fbase), B_m, H_m,
                                facecolor="#F1F5F9", edgecolor="#0F172A", lw=2))

    col_top = Df_m + 0.4
    col_x_sec = (B_m - col_bx/100)/2
    ax2.add_patch(plt.Rectangle((col_x_sec, fbase+H_m), col_bx/100, col_top-(fbase+H_m),
                                facecolor="#FEE2E2", edgecolor="#0F172A", lw=1.5))

    rx1, rx2 = cover_m, B_m - cover_m
    ry = fbase + cover_m
    ax2.plot([rx1, rx2], [ry, ry], color="#1D4ED8", lw=2.5)
    ax2.plot([rx1, rx1], [ry, ry+0.12], color="#1D4ED8", lw=2.5)
    ax2.plot([rx2, rx2], [ry, ry+0.12], color="#1D4ED8", lw=2.5)

    n_dots = min(bars_y, 15)
    for i in range(n_dots):
        dx = rx1 + i * (rx2-rx1) / max(n_dots-1, 1)
        ax2.plot(dx, ry+0.02, "o", markersize=4, color="#DC2626")

    dw1, dw2 = col_x_sec + 0.05, col_x_sec + col_bx/100 - 0.05
    ax2.plot([dw1, dw1], [ry+0.02, col_top+0.1], color="#047857", lw=2)
    ax2.plot([dw2, dw2], [ry+0.02, col_top+0.1], color="#047857", lw=2)
    ax2.plot([dw1, dw1-0.1], [ry+0.02, ry+0.02], color="#047857", lw=2)
    ax2.plot([dw2, dw2+0.1], [ry+0.02, ry+0.02], color="#047857", lw=2)

    draw_dim(ax2, 0, -0.15, B_m, -0.15, f"B = {B_m:.2f} m", off_y=-0.05)
    ax2.plot([-0.2, -0.1], [fbase, fbase], color="#475569", lw=1)
    ax2.plot([-0.2, -0.1], [fbase+H_m, fbase+H_m], color="#475569", lw=1)
    ax2.plot([-0.15, -0.15], [fbase, fbase+H_m], color="#475569", lw=1)
    ax2.text(-0.25, fbase+H_m/2, f"H={H_cm:.0f}cm", ha="center", va="center",
             color="#0F172A", fontsize=9, rotation=90)

    ax2.annotate(f"{bars_x}×DB16 Main", xy=(rx2-0.2, ry), xytext=(rx2+0.15, ry-0.2),
                 arrowprops=dict(arrowstyle="->", lw=1), fontsize=8)
    ax2.annotate(f"{bars_y}×DB16 Cross", xy=(rx1+0.2, ry+0.02), xytext=(rx1-0.35, ry-0.2),
                 arrowprops=dict(arrowstyle="->", lw=1), fontsize=8)

    ax2.set_xlim(-0.6, B_m+0.6)
    ax2.set_ylim(-0.35, Df_m+0.65)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_title("SECTION ELEVATION", fontsize=12, fontweight="bold", color="#1E3A8A", pad=15)
    ax2.axis("off")

    st.pyplot(fig)
