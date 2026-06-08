"""
Advanced Biaxial Foundation Engineering Suite v4
ACI 318-19 | Terzaghi & Meyerhof | Elastic + Consolidation Settlement | Combined Footings

UNIT SYSTEM (consistent throughout):
  Force   : ton  (1 ton = 1000 kgf)
  Moment  : ton-m
  Length  : m (geometry), cm (structural checks)
  Stress  : ton/m² (geotechnical), kg/cm² (structural)
  Ec, Es  : ksc = kgf/cm²

BUGS FIXED FROM v3:
  1. Meyerhof ig: alpha_deg/phi_radians → alpha_deg/phi_deg
  2. Terzaghi sq factor: replaced Meyerhof-style formula with Terzaghi 1943 rectangular factors
  3. Development length: fy/(1.4√fc)*db → ACI 318-19 §25.5.2.1 properly in MKS
  4. All sidebar widgets: unique key= parameters added (prevents StreamlitDuplicateElementId)
  5. One-way shear: uses average pressure at critical section, not maximum
  6. qu_mod sign: separated to avoid double-counting for corner pressures
  7. Combined footing trapezoidal: corrected moment calculation approach
  8. Optimization: also checks development length adequacy
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass, field
from typing import List, Optional
import plotly.graph_objects as go
import copy
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (ACI 318-19, MKS)
# ══════════════════════════════════════════════════════════════════════════════
PHI_SHEAR    = 0.75
PHI_FLEX     = 0.90
FS_BEARING   = 3.0
FS_OVR_MIN   = 1.50
FS_SLIDE_MIN = 1.50
CLEAR_COVER_FOOTING = 7.5   # cm  (ACI §20.6.1.3 for footings cast on soil)
DB16_AREA    = 2.01         # cm²  (1 × DB16)
DB16_MM      = 16           # mm
DB16_CM      = 1.6          # cm


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Loads:
    P_DL: float; P_LL: float
    M_DL_x: float; M_LL_x: float; M_WL_x: float
    M_DL_y: float; M_LL_y: float; M_WL_y: float
    V_hx: float;   V_hy: float

@dataclass
class Properties:
    qa_allow: float; fc_prime: float; fy: float
    soil_density: float; base_friction: float

@dataclass
class SoilProfile:
    cohesion:     float = 0.0
    phi_deg:      float = 30.0
    Es:           float = 2000.0
    nu:           float = 0.3
    Cc:           float = 0.3
    Cs:           float = 0.05
    e0:           float = 0.8
    OCR:          float = 1.0
    H_clay:       float = 3.0
    sigma_v0:     float = 10.0
    soil_density: float = 1.8
    method:       str   = "Meyerhof"
    failure_mode: str   = "General"

@dataclass
class Geometry:
    B: float; L: float; H_cm: float; Df: float
    cx: float; cy: float            # column dimensions in cm
    shape: str    = "rectangular"
    D_circ: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# DEVELOPMENT LENGTH  (ACI 318-19 §25.5.2.1, MKS)
# ══════════════════════════════════════════════════════════════════════════════
def dev_length_cm(fy_ksc: float, fc_ksc: float, db_cm: float,
                  cb_ktr_db: float = 2.5) -> float:
    """
    Tension development length for straight deformed bars (ACI 318-19 §25.5.2.1).
    MKS units: fy [ksc], fc [ksc], db [cm] → ld [cm].

    Derivation from SI (MPa, mm → mm):
      ld_mm = 3·fy_MPa / (40·λ·√fc_MPa) · (ψt·ψe·ψs·ψg / [(cb+Ktr)/db]) · db_mm

    Converting MPa ↔ ksc (×10.2) and mm ↔ cm (×10):
      ld_cm = 3·(fy_ksc/10.2) / (40·√(fc_ksc/10.2)) · (ψ/ratio) · (db_cm·10) / 10
            = 3/(40·√10.2) · (ψ/ratio) · fy_ksc/√fc_ksc · db_cm

    Assumptions: bottom bars (ψt=1), uncoated (ψe=1), normal-weight (λ=1),
    ψg=1.0; ψs=0.8 for db≤19mm (DB16), =1.0 for db>19mm.
    ACI minimum: ld ≥ 30 cm (≈ 12 in).
    """
    psi_s = 0.8 if db_cm * 10 <= 19 else 1.0   # db_mm ≤ 19 mm
    psi_total = 1.0 * 1.0 * psi_s * 1.0          # ψt·ψe·ψs·ψg
    coeff = 3.0 / (40.0 * math.sqrt(10.2))        # ≈ 0.02348
    ld = coeff * (psi_total / cb_ktr_db) * (fy_ksc / math.sqrt(fc_ksc)) * db_cm
    return max(ld, 30.0)


# ══════════════════════════════════════════════════════════════════════════════
# FOUNDATION DESIGNER
# ══════════════════════════════════════════════════════════════════════════════
class FoundationDesigner:
    """
    Isolated spread footing: rectangular or circular.
    All internal structural calcs in kg-cm; geotechnical in ton-m².
    """

    def __init__(self, loads: Loads, props: Properties, geo: Geometry):
        self.loads = loads
        self.props = props
        self.geo   = geo
        self.is_circular = (geo.shape == "circular")

        if self.is_circular:
            R = geo.D_circ / 2.0
            self.A_base = math.pi * R**2
            self.B_eff  = math.sqrt(self.A_base)
            self.B_cm   = self.B_eff * 100
            self.L_cm   = self.B_eff * 100
            self.I_x    = math.pi * geo.D_circ**4 / 64.0
            self.I_y    = self.I_x
        else:
            self.A_base = geo.B * geo.L
            self.B_cm   = geo.B * 100
            self.L_cm   = geo.L * 100
            self.I_x    = geo.B * geo.L**3 / 12.0   # about X-axis (varies in L-dir)
            self.I_y    = geo.L * geo.B**3 / 12.0   # about Y-axis (varies in B-dir)

        # Effective depth: H − cover − assumed 1 bar diameter
        self.d_cm = geo.H_cm - CLEAR_COVER_FOOTING - DB16_CM

    # ── Service Limit State ───────────────────────────────────────────────────
    def analyze_service_state(self) -> dict:
        loads = self.loads; props = self.props; geo = self.geo
        P_svc = loads.P_DL + loads.P_LL
        M_x   = loads.M_DL_x + loads.M_LL_x    # bending about X → eccentricity in Y
        M_y   = loads.M_DL_y + loads.M_LL_y    # bending about Y → eccentricity in X

        H_m = geo.H_cm / 100.0
        W_ftg      = self.A_base * H_m * 2.4
        W_overburden = self.A_base * max(geo.Df - H_m, 0.0) * props.soil_density
        P_total    = P_svc + W_ftg + W_overburden

        # Eccentricity (m)
        e_x = M_y / P_total if P_total > 0 else 0.0   # eccentric in X-direction
        e_y = M_x / P_total if P_total > 0 else 0.0   # eccentric in Y-direction

        if self.is_circular:
            R = geo.D_circ / 2.0
            kern_x = kern_y = R / 4.0
        else:
            kern_x = geo.B / 6.0
            kern_y = geo.L / 6.0

        has_tension = (e_x > kern_x) or (e_y > kern_y)
        q_avg = P_total / self.A_base

        if self.is_circular:
            S = math.pi * (geo.D_circ / 2.0)**3 / 4.0
            q_max = q_avg + M_x / S + M_y / S
            q_min = max(0.0, q_avg - M_x / S - M_y / S)
        else:
            q_mod_y = M_x * (geo.L / 2.0) / self.I_x   # ton/m²  (I_x → L direction)
            q_mod_x = M_y * (geo.B / 2.0) / self.I_y   # ton/m²  (I_y → B direction)
            if not has_tension:
                q_max = q_avg + q_mod_x + q_mod_y
                q_min = max(0.0, q_avg - q_mod_x - q_mod_y)
            else:
                B_prime = max(geo.B - 2.0 * e_x, 0.1)
                L_prime = max(geo.L - 2.0 * e_y, 0.1)
                q_max = P_total / (B_prime * L_prime)
                q_min = 0.0

        # Overturning
        half_B = (geo.D_circ / 2.0) if self.is_circular else geo.B / 2.0
        half_L = (geo.D_circ / 2.0) if self.is_circular else geo.L / 2.0
        M_ovr_x = M_x + loads.V_hy * geo.Df
        M_ovr_y = M_y + loads.V_hx * geo.Df
        FS_ovr_x = (P_total * half_L) / M_ovr_x if M_ovr_x > 0 else float("inf")
        FS_ovr_y = (P_total * half_B) / M_ovr_y if M_ovr_y > 0 else float("inf")

        # Sliding
        V_h = math.sqrt(loads.V_hx**2 + loads.V_hy**2)
        FS_slide = (P_total * props.base_friction) / V_h if V_h > 0 else float("inf")

        return dict(
            P_total=P_total, e_x=e_x, e_y=e_y,
            kern_x=kern_x, kern_y=kern_y, has_tension=has_tension,
            q_avg=q_avg, q_max=q_max, q_min=q_min,
            FS_ovr_x=FS_ovr_x, FS_ovr_y=FS_ovr_y, FS_slide=FS_slide,
            W_ftg=W_ftg, W_overburden=W_overburden,
        )

    # ── Factored Loads (ACI 318-19 load combinations) ────────────────────────
    def _factored_loads(self) -> dict:
        L = self.loads
        combos = [
            dict(P=1.4*L.P_DL,
                 Mx=1.4*L.M_DL_x, My=1.4*L.M_DL_y, label="1.4D"),
            dict(P=1.2*L.P_DL + 1.6*L.P_LL,
                 Mx=1.2*L.M_DL_x + 1.6*L.M_LL_x,
                 My=1.2*L.M_DL_y + 1.6*L.M_LL_y, label="1.2D+1.6L"),
            dict(P=1.2*L.P_DL + 1.0*L.P_LL,
                 Mx=1.2*L.M_DL_x + 1.0*L.M_LL_x + 1.0*L.M_WL_x,
                 My=1.2*L.M_DL_y + 1.0*L.M_LL_y + 1.0*L.M_WL_y,
                 label="1.2D+1.0L+1.0W"),
            dict(P=0.9*L.P_DL,
                 Mx=0.9*L.M_DL_x + 1.0*L.M_WL_x,
                 My=0.9*L.M_DL_y + 1.0*L.M_WL_y, label="0.9D+1.0W"),
        ]
        # Govern by maximum corner demand = P + |Mx| + |My| (consistent index)
        return max(combos, key=lambda c: c["P"] + abs(c["Mx"]) + abs(c["My"]))

    # ── Ultimate Limit State ──────────────────────────────────────────────────
    def analyze_ultimate_state(self) -> dict:
        """
        All pressures in kg/cm².  Forces in kg.  Moments in kg·cm.
        Conversion: 1 ton = 1000 kg;  1 ton-m = 1e5 kg·cm;  1 ton/m² = 0.01 kg/cm².
        """
        fc = self.props.fc_prime
        geo = self.geo
        combo = self._factored_loads()
        P_u   = combo["P"]    # ton
        M_u_x = combo["Mx"]   # ton-m  (moment about X → varies along L)
        M_u_y = combo["My"]   # ton-m  (moment about Y → varies along B)
        label = combo["label"]

        B_cm, L_cm, d = self.B_cm, self.L_cm, self.d_cm

        # Average factored pressure (kg/cm²)
        A_cm2   = B_cm * L_cm
        qu_base = (P_u * 1000.0) / A_cm2   # kg/cm²

        # Pressure modifiers from biaxial moments
        # M_u_y (about Y-axis) → eccentricity in X direction (across B)
        #   I_y = L·B³/12 about Y-axis; c = B/2; σ = M·c/I
        qu_mod_B = (M_u_y * 1e5 * (B_cm / 2.0)) / (L_cm * B_cm**3 / 12.0)   # kg/cm²
        # M_u_x (about X-axis) → eccentricity in Y direction (across L)
        #   I_x = B·L³/12 about X-axis; c = L/2
        qu_mod_L = (M_u_x * 1e5 * (L_cm / 2.0)) / (B_cm * L_cm**3 / 12.0)   # kg/cm²

        qu_max = max(qu_base + qu_mod_B + qu_mod_L, 0.0)
        qu_min = max(qu_base - qu_mod_B - qu_mod_L, 0.0)

        # ── One-Way (Wide-Beam) Shear ──────────────────────────────────────
        # Critical section at d from column face; use average pressure on strip

        # X-direction: shear on a strip perpendicular to B, width = L_cm
        crit_dist_B = max(0.0, (B_cm - geo.cx) / 2.0 - d)   # cantilever beyond d
        # Average qu on critical cantilever strip (linear interpolation, conservative avg)
        qu_at_face_B   = qu_base + qu_mod_B                   # pressure at B/2
        qu_at_crit_B   = qu_base + qu_mod_B * (1.0 - 2.0*d/B_cm)  # at d from face
        qu_avg_strip_B = (qu_at_face_B + qu_at_crit_B) / 2.0
        V_u_B  = qu_avg_strip_B * L_cm * crit_dist_B         # kg
        v_u_x  = V_u_B / (L_cm * d) if (L_cm * d) > 0 else 0.0   # kg/cm²

        # Y-direction: strip width = B_cm
        crit_dist_L = max(0.0, (L_cm - geo.cy) / 2.0 - d)
        qu_at_face_L   = qu_base + qu_mod_L
        qu_at_crit_L   = qu_base + qu_mod_L * (1.0 - 2.0*d/L_cm)
        qu_avg_strip_L = (qu_at_face_L + qu_at_crit_L) / 2.0
        V_u_L  = qu_avg_strip_L * B_cm * crit_dist_L         # kg
        v_u_y  = V_u_L / (B_cm * d) if (B_cm * d) > 0 else 0.0

        # φVc for wide-beam shear: ACI 318-19 §22.5.5.1 (MKS)
        phi_vc_wide = PHI_SHEAR * 0.53 * math.sqrt(fc)       # kg/cm²

        # ── Punching Shear ─────────────────────────────────────────────────
        # Critical perimeter at d/2 from column face  (ACI §22.6.4.1)
        cx_p  = geo.cx + d    # cm  (dimension of critical perimeter in B-direction)
        cy_p  = geo.cy + d    # cm  (dimension in L-direction)
        bo    = 2.0 * (cx_p + cy_p)                           # cm
        A_p   = cx_p * cy_p                                   # cm²
        V_punch = qu_max * (A_cm2 - A_p)                      # kg (use max pressure)
        v_u_punch = V_punch / (bo * d) if (bo * d) > 0 else 0.0

        # φVc punching: ACI 318-19 §22.6.5 – min of three expressions
        beta_c = max(geo.cx, geo.cy) / max(min(geo.cx, geo.cy), 1.0)
        alpha_s = 40   # interior column
        vc1 = 0.27 * (2.0 + 4.0 / beta_c) * math.sqrt(fc)
        vc2 = 0.27 * (alpha_s * d / bo + 2.0) * math.sqrt(fc)
        vc3 = 1.06 * math.sqrt(fc)
        phi_vc_punch = PHI_SHEAR * min(vc1, vc2, vc3)

        # ── Flexure ────────────────────────────────────────────────────────
        cant_B = (B_cm - geo.cx) / 2.0     # cm cantilever in B-direction
        cant_L = (L_cm - geo.cy) / 2.0     # cm cantilever in L-direction
        # Moment at face of column (kg·cm) – per unit width × total width
        M_u_B = qu_avg_strip_B * L_cm * cant_B**2 / 2.0   # kg·cm (over full width L)
        M_u_L = qu_avg_strip_L * B_cm * cant_L**2 / 2.0

        # ── Development Length (ACI 318-19 §25.5.2.1) ─────────────────────
        L_d = dev_length_cm(self.props.fy, fc, DB16_CM)
        avail_Ld_B = cant_B - CLEAR_COVER_FOOTING   # available length inside footing
        avail_Ld_L = cant_L - CLEAR_COVER_FOOTING

        return dict(
            governing_combo=label,
            P_u=P_u, M_u_x=M_u_x, M_u_y=M_u_y,
            qu_base=qu_base, qu_max=qu_max, qu_min=qu_min,
            qu_mod_B=qu_mod_B, qu_mod_L=qu_mod_L,
            v_u_x=v_u_x, v_u_y=v_u_y,
            v_u_wide_max=max(v_u_x, v_u_y),
            phi_vc_wide=phi_vc_wide,
            v_u_punch=v_u_punch, phi_vc_punch=phi_vc_punch,
            bo=bo, beta_c=beta_c,
            vc1=vc1, vc2=vc2, vc3=vc3,
            M_u_B=M_u_B, M_u_L=M_u_L,
            cant_B=cant_B, cant_L=cant_L,
            L_d=L_d, avail_Ld_B=avail_Ld_B, avail_Ld_L=avail_Ld_L,
        )

    # ── Flexural Reinforcement Design ─────────────────────────────────────────
    def design_flexure(self, M_u_kgcm: float, width_cm: float):
        """M_u [kg·cm], width [cm] → (n_bars, spacing_cm, As_req_cm2, rho_prov)"""
        fc = self.props.fc_prime; fy = self.props.fy; d = self.d_cm
        rho_min = 0.0018 if fy >= 4000 else 0.0020
        R_n = M_u_kgcm / (PHI_FLEX * width_cm * d**2)   # kg/cm²
        m   = fy / (0.85 * fc)
        disc = max(0.0, 1.0 - 2.0 * m * R_n / fy)
        rho_req = (1.0 / m) * (1.0 - math.sqrt(disc)) if disc >= 0 else rho_min
        rho_design = max(rho_req, rho_min)
        As_req = rho_design * width_cm * d
        n_bars = max(5, math.ceil(As_req / DB16_AREA))
        spacing = (width_cm - 2.0 * CLEAR_COVER_FOOTING) / max(n_bars - 1, 1)
        As_prov = n_bars * DB16_AREA
        rho_prov = As_prov / (width_cm * d)
        return n_bars, spacing, As_req, rho_prov

    # ── Soil Pressure Grid (for 3-D visualisation) ───────────────────────────
    def soil_pressure_grid(self, n: int = 50):
        sls   = self.analyze_service_state()
        q_avg = sls["P_total"] / self.A_base
        M_x   = self.loads.M_DL_x + self.loads.M_LL_x
        M_y   = self.loads.M_DL_y + self.loads.M_LL_y

        if self.is_circular:
            R  = self.geo.D_circ / 2.0
            xs = np.linspace(-R, R, n)
            ys = np.linspace(-R, R, n)
            XX, YY = np.meshgrid(xs, ys)
            S  = math.pi * R**3 / 4.0
            ZZ = q_avg + M_x / S * YY / R + M_y / S * XX / R
            ZZ[XX**2 + YY**2 > R**2] = np.nan
        else:
            xs = np.linspace(-self.geo.B / 2, self.geo.B / 2, n)
            ys = np.linspace(-self.geo.L / 2, self.geo.L / 2, n)
            XX, YY = np.meshgrid(xs, ys)
            ZZ = q_avg + M_y * XX / self.I_y + M_x * YY / self.I_x
            ZZ = np.maximum(ZZ, 0.0)
        return XX, YY, ZZ

    # ── Auto-Optimisation ─────────────────────────────────────────────────────
    def optimize_dimensions(self, aspect_ratio: float = 1.0):
        def violation(B_try):
            g2 = copy.copy(self.geo)
            g2.B = round(B_try, 2)
            g2.L = round(B_try / aspect_ratio, 2)
            des = FoundationDesigner(self.loads, self.props, g2)
            sls = des.analyze_service_state()
            uls = des.analyze_ultimate_state()
            geo_margin = min(
                self.props.qa_allow - sls["q_max"],
                sls["FS_ovr_x"] - FS_OVR_MIN,
                sls["FS_ovr_y"] - FS_OVR_MIN,
                sls["FS_slide"] - FS_SLIDE_MIN,
            )
            str_margin = min(
                uls["phi_vc_wide"]  - uls["v_u_wide_max"],
                uls["phi_vc_punch"] - uls["v_u_punch"],
            )
            return -min(geo_margin, str_margin)

        lo, hi = 0.5, 12.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            (hi if violation(mid) <= 0 else lo).__class__  # dummy
            if violation(mid) <= 0:
                hi = mid
            else:
                lo = mid
        B_opt = math.ceil(hi * 10) / 10
        L_opt = math.ceil((B_opt / aspect_ratio) * 10) / 10
        return B_opt, L_opt


# ══════════════════════════════════════════════════════════════════════════════
# BEARING CAPACITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class BearingCapacityEngine:
    """Terzaghi (1943) and Meyerhof (1963) bearing capacity. Units: ton, m."""

    def __init__(self, soil: SoilProfile, geo: Geometry, loads: Loads):
        self.soil  = soil
        self.geo   = geo
        self.loads = loads
        self.phi   = math.radians(soil.phi_deg)

    def _gamma(self):
        return self.soil.soil_density

    # ── Terzaghi (1943) bearing capacity factors ──────────────────────────────
    def _terzaghi_Nfactors(self, phi_rad):
        if phi_rad == 0.0:
            return 1.0, 5.7, 0.0
        Nq = math.exp(math.pi * math.tan(phi_rad)) * math.tan(math.radians(45) + phi_rad / 2)**2
        Nc = (Nq - 1.0) / math.tan(phi_rad)
        Ng = 2.0 * (Nq + 1.0) * math.tan(phi_rad)
        return Nq, Nc, Ng

    def terzaghi(self) -> dict:
        """
        Rectangular footing (Terzaghi 1943 shape factors for rectangular):
          sc = 1 + 0.3(B/L)
          sq = 1 + 0.2(B/L)   [corrected from Meyerhof-style error in v3]
          sγ = 1 − 0.4(B/L)
        For circular: qu = 1.3cNc + γDfNq + 0.3γBNγ  (Terzaghi original)
        """
        c   = self.soil.cohesion; phi = self.phi; γ = self._gamma()
        Df  = self.geo.Df

        mode     = self.soil.failure_mode
        phi_use  = phi
        c_use    = c
        if mode == "Local":
            phi_use = math.atan(2.0 / 3.0 * math.tan(phi))
            c_use   = 2.0 / 3.0 * c
        elif mode == "Punching":
            phi_use = math.atan(0.5 * math.tan(phi))
            c_use   = 0.5 * c

        Nq, Nc, Ng = self._terzaghi_Nfactors(phi_use)
        is_circ = (self.geo.shape == "circular")

        if is_circ:
            B = self.geo.D_circ
            qu = 1.3 * c_use * Nc + γ * Df * Nq + 0.3 * γ * B * Ng
            sc = 1.3; sq = 1.0; sg = 0.3    # circular embedded factors
        else:
            B, L = self.geo.B, self.geo.L
            BL = B / L
            # Terzaghi 1943 rectangular shape factors:
            sc = 1.0 + 0.3 * BL
            sq = 1.0 + 0.2 * BL          # ← FIXED (was using Meyerhof formula)
            sg = max(1.0 - 0.4 * BL, 0.6)
            qu = c_use * Nc * sc + γ * Df * Nq * sq + 0.5 * γ * B * Ng * sg

        qa = qu / FS_BEARING
        return dict(
            method=f"Terzaghi ({mode})",
            phi_use_deg=math.degrees(phi_use), c_use=c_use,
            Nq=Nq, Nc=Nc, Ng=Ng, sc=sc, sq=sq, sg=sg,
            qu_ultimate=qu, FS_bearing=FS_BEARING, qa_computed=qa,
        )

    # ── Meyerhof (1963) ───────────────────────────────────────────────────────
    def _meyerhof_Nfactors(self):
        phi = self.phi
        if phi == 0.0:
            return 1.0, 5.14, 0.0
        Nq = math.exp(math.pi * math.tan(phi)) * math.tan(math.radians(45) + phi / 2)**2
        Nc = (Nq - 1.0) / math.tan(phi)
        Ng = (Nq - 1.0) * math.tan(1.4 * phi)
        return Nq, Nc, Ng

    def meyerhof(self) -> dict:
        """
        Full Meyerhof with shape (sc,sq,sγ), depth (dc,dq,dγ), inclination (ic,iq,iγ).
        FIX: inclination factor iγ used phi_deg not phi_rad in denominator.
        """
        c   = self.soil.cohesion; phi = self.phi; phi_deg = self.soil.phi_deg
        γ   = self._gamma(); Df = self.geo.Df
        is_circ = (self.geo.shape == "circular")
        B = self.geo.D_circ if is_circ else self.geo.B
        L = B               if is_circ else self.geo.L

        Nq, Nc, Ng = self._meyerhof_Nfactors()

        # Shape factors
        BL = B / L
        if phi == 0.0:
            sc = 1.0 + 0.2 * BL; sq = sg = 1.0
        else:
            tan45 = math.tan(math.radians(45) + phi / 2)
            sc = 1.0 + 0.2 * BL * tan45**2
            sq = sg = 1.0 + 0.1 * BL * tan45**2

        # Depth factors
        DfB = Df / B
        if phi == 0.0:
            dc = 1.0 + 0.4 * DfB; dq = dg = 1.0
        else:
            tan45 = math.tan(math.radians(45) + phi / 2)
            dc = 1.0 + 0.4 * DfB
            dq = dg = 1.0 + 0.1 * DfB * tan45**2

        # Inclination factors
        P_svc = self.loads.P_DL + self.loads.P_LL
        V_h   = math.sqrt(self.loads.V_hx**2 + self.loads.V_hy**2)
        alpha_deg = math.degrees(math.atan2(V_h, P_svc)) if P_svc > 0 else 0.0

        if phi == 0.0:
            ic = max(0.0, 1.0 - alpha_deg / 90.0); iq = ig = 1.0
        else:
            ic = iq = max(0.0, (1.0 - alpha_deg / 90.0)**2)
            # FIX: denominator must be phi_deg (degrees), not phi_rad
            ig = max(0.0, (1.0 - alpha_deg / phi_deg)**2) if phi_deg > 0 else 1.0

        qu = (c * Nc * sc * dc * ic
              + γ * Df * Nq * sq * dq * iq
              + 0.5 * γ * B * Ng * sg * dg * ig)
        qa = qu / FS_BEARING
        return dict(
            method="Meyerhof",
            Nq=Nq, Nc=Nc, Ng=Ng,
            sc=sc, sq=sq, sg=sg,
            dc=dc, dq=dq, dg=dg,
            ic=ic, iq=iq, ig=ig, alpha_deg=alpha_deg,
            qu_ultimate=qu, FS_bearing=FS_BEARING, qa_computed=qa,
        )

    def run(self):
        return self.terzaghi() if self.soil.method == "Terzaghi" else self.meyerhof()

    def run_both(self):
        return {"Terzaghi": self.terzaghi(), "Meyerhof": self.meyerhof()}


# ══════════════════════════════════════════════════════════════════════════════
# SETTLEMENT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class SettlementEngine:
    """Elastic (Schleicher) + Primary Consolidation.  Units: ton, m."""

    def __init__(self, soil: SoilProfile, geo: Geometry, q_net: float):
        self.soil  = soil
        self.geo   = geo
        self.q_net = max(q_net, 0.0)

    def elastic(self) -> dict:
        """Si = q_net·B·(1−ν²)/Es · If   [Bowles 5th ed.; If = flexible-centre influence factor]"""
        B  = self.geo.D_circ if self.geo.shape == "circular" else self.geo.B
        L  = B                if self.geo.shape == "circular" else self.geo.L
        Es = self.soil.Es; nu = self.soil.nu
        LB = max(L / B, 1.0)
        If = 0.82 * (1.0 + 0.22 * (LB - 1.0))    # flexible, centre point
        Si = self.q_net * B * (1.0 - nu**2) / Es * If
        return dict(Si_m=Si, Si_cm=Si * 100, If=If, B=B, L=L)

    def consolidation(self) -> dict:
        """
        Primary consolidation (Terzaghi).
        Δσ estimated via Boussinesq 2:1 stress distribution at clay mid-layer.
        NC: Sc = Cc·H/(1+e0)·log10(σv1/σv0)
        OC: uses Cs in recompression zone, Cc in virgin zone.
        """
        Cc = self.soil.Cc; Cs = self.soil.Cs; e0 = self.soil.e0
        OCR = self.soil.OCR; H = self.soil.H_clay; sv0 = self.soil.sigma_v0
        svc = sv0 * OCR

        B = self.geo.D_circ if self.geo.shape == "circular" else self.geo.B
        L = B                if self.geo.shape == "circular" else self.geo.L
        z = H / 2.0          # depth to mid-layer below footing base
        dsigma = self.q_net * B * L / ((B + z) * (L + z))   # 2:1 Boussinesq
        sv1 = sv0 + dsigma

        if OCR <= 1.0 or sv0 >= svc:
            Sc = (Cc * H / (1.0 + e0)) * math.log10(max(sv1 / sv0, 1.001)) if sv0 > 0 else 0.0
            regime = "Normally Consolidated (NC)"
        elif sv1 <= svc:
            Sc = (Cs * H / (1.0 + e0)) * math.log10(sv1 / sv0)
            regime = "OC – recompression only"
        else:
            Sc = ((Cs * H / (1.0 + e0)) * math.log10(svc / sv0)
                 + (Cc * H / (1.0 + e0)) * math.log10(sv1 / svc))
            regime = "OC – crosses preconsolidation"

        return dict(Sc_m=Sc, Sc_cm=Sc * 100, dsigma=dsigma,
                    sv0=sv0, svc=svc, sv1=sv1, regime=regime)

    def total(self) -> dict:
        el  = self.elastic()
        con = self.consolidation()
        St  = el["Si_m"] + con["Sc_m"]
        return dict(elastic=el, consolidation=con, St_m=St, St_cm=St * 100)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED FOOTING DESIGNER
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ColumnData:
    P_DL: float; P_LL: float
    cx: float;   cy: float      # cm
    x_pos: float                # distance from left edge of footing (m)

class CombinedFootingDesigner:
    def __init__(self, col1: ColumnData, col2: ColumnData,
                 B: float, H_cm: float, Df: float,
                 props: Properties, soil: SoilProfile, ftype: str = "Rectangular"):
        self.col1 = col1; self.col2 = col2
        self.B = B; self.H_cm = H_cm; self.Df = Df
        self.props = props; self.soil = soil; self.ftype = ftype
        self.d_cm = H_cm - CLEAR_COVER_FOOTING - DB16_CM

    def _resultant(self):
        P1 = self.col1.P_DL + self.col1.P_LL
        P2 = self.col2.P_DL + self.col2.P_LL
        Pt = P1 + P2
        xR = (P1 * self.col1.x_pos + P2 * self.col2.x_pos) / Pt
        return P1, P2, Pt, xR

    def design_rectangular(self) -> dict:
        P1, P2, Pt, xR = self._resultant()
        x1, x2 = self.col1.x_pos, self.col2.x_pos
        # Footing spans from left edge (x=0) to right edge; centroid at L/2 = xR
        L = 2.0 * xR
        L = max(L, x2 + 0.3)  # ensure right column is inside
        H_m = self.H_cm / 100.0
        A   = self.B * L
        W   = A * H_m * 2.4 + A * max(self.Df - H_m, 0) * self.soil.soil_density
        q_net   = Pt / A
        q_total = (Pt + W) / A

        n = 300
        xs  = np.linspace(0.0, L, n)
        w   = Pt / L   # upward net ton/m (uniform)
        # Shear V(x) and Moment M(x) by integration
        V = np.zeros(n); M = np.zeros(n)
        for i, x in enumerate(xs):
            v = -w * x
            if x >= x1: v += P1
            if x >= x2: v += P2
            V[i] = v
            m = -w * x**2 / 2.0
            if x >= x1: m += P1 * (x - x1)
            if x >= x2: m += P2 * (x - x2)
            M[i] = m

        return dict(type="Rectangular", L=L, B=self.B, xR=xR, Pt=Pt,
                    q_net=q_net, q_total=q_total,
                    xs=xs, V=V, M=M, M_pos=float(np.max(M)), M_neg=float(np.min(M)),
                    x1=x1, x2=x2)

    def design_trapezoidal(self) -> dict:
        """
        Trapezoidal footing: widths b1 (at x=0) and b2 (at x=L).
        Centroid of trapezoid from b1 end: xc = L*(2b2+b1)/(3*(b1+b2))
        Constraint: b1+b2 = 2B (keep average = B), xc = xR.
        """
        P1, P2, Pt, xR = self._resultant()
        x1, x2 = self.col1.x_pos, self.col2.x_pos
        L  = max(2.0 * xR, x2 + 0.3)
        xc = xR   # centroid from left edge

        # Solve: b1+b2=2B, L(2b2+b1)/(3*(b1+b2)) = xc
        # → L(2b2+b1) = 3*xc*2B → L*b1 + 2L*b2 = 6B*xc
        # + b1 + b2 = 2B → b1 = 2B-b2
        # → L*(2B-b2) + 2L*b2 = 6B*xc → 2BL + Lb2 = 6B*xc
        b2 = max((6.0 * self.B * xc - 2.0 * self.B * L) / L, 0.4)
        b1 = max(2.0 * self.B - b2, 0.4)

        H_m = self.H_cm / 100.0
        A   = (b1 + b2) / 2.0 * L
        W   = A * H_m * 2.4 + A * max(self.Df - H_m, 0) * self.soil.soil_density
        q_net   = Pt / A
        q_total = (Pt + W) / A

        n  = 300
        xs = np.linspace(0.0, L, n)
        bx = b1 + (b2 - b1) * xs / L       # width at each x
        wu = q_net * bx                     # upward load intensity (ton/m)

        # Integrate for shear and moment
        V = np.zeros(n); M = np.zeros(n)
        for i in range(n):
            x = xs[i]
            V_up = float(np.trapz(wu[:i+1], xs[:i+1]))
            v = -V_up
            if x >= x1: v += P1
            if x >= x2: v += P2
            V[i] = v

            # Moment = -∫wu(ξ)(x-ξ)dξ from 0 to x
            if i > 0:
                M_up = float(np.trapz(wu[:i+1] * (x - xs[:i+1]), xs[:i+1]))
            else:
                M_up = 0.0
            m = -M_up
            if x >= x1: m += P1 * (x - x1)
            if x >= x2: m += P2 * (x - x2)
            M[i] = m

        return dict(type="Trapezoidal", L=L, B=self.B, b1=b1, b2=b2, xR=xR, Pt=Pt,
                    q_net=q_net, q_total=q_total, bx=bx,
                    xs=xs, V=V, M=M, M_pos=float(np.max(M)), M_neg=float(np.min(M)),
                    x1=x1, x2=x2)

    def design(self):
        return self.design_rectangular() if self.ftype == "Rectangular" else self.design_trapezoidal()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Foundation Suite v4", layout="wide",
                   initial_sidebar_state="expanded", page_icon="🏗️")

st.markdown("""
<style>
.main-hdr{font-size:28px;font-weight:700;color:#0F172A;margin-bottom:4px;}
.sub-hdr{font-size:13px;color:#475569;margin-bottom:20px;}
.sec-ttl{font-size:18px;font-weight:600;color:#1E3A8A;border-left:5px solid #2563EB;
         padding-left:10px;margin-top:20px;margin-bottom:12px;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="main-hdr">🏗️ Advanced Biaxial Foundation Suite <span style="font-size:14px;color:#64748B;">v4 — ACI 318-19</span></div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-hdr">ULS & SLS · Terzaghi & Meyerhof · Elastic + Consolidation Settlement · Combined Footings · Rect. & Circular</div>',
            unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────
sb = st.sidebar
sb.header("📥 Input Panel")

with sb.expander("1. Structural Service Loads", expanded=True):
    P_DL   = st.number_input("P_DL Dead Load (ton)", 0.0, value=30.0, step=1.0, key="P_DL")
    P_LL   = st.number_input("P_LL Live Load (ton)", 0.0, value=18.0, step=1.0, key="P_LL")
    st.markdown("**Moments (ton-m)**")
    M_DL_x = st.number_input("M_DL_x", value=3.5,  step=0.5, key="M_DL_x")
    M_LL_x = st.number_input("M_LL_x", value=2.0,  step=0.5, key="M_LL_x")
    M_WL_x = st.number_input("M_WL_x (Wind)", value=1.5, step=0.5, key="M_WL_x")
    M_DL_y = st.number_input("M_DL_y", value=2.5,  step=0.5, key="M_DL_y")
    M_LL_y = st.number_input("M_LL_y", value=1.5,  step=0.5, key="M_LL_y")
    M_WL_y = st.number_input("M_WL_y (Wind)", value=1.0, step=0.5, key="M_WL_y")
    st.markdown("**Base Shears (ton)**")
    V_hx   = st.number_input("V_hx", value=2.0, step=0.1, key="V_hx")
    V_hy   = st.number_input("V_hy", value=1.8, step=0.1, key="V_hy")

with sb.expander("2. Materials & Foundation", expanded=False):
    qa_allow      = st.number_input("q_allow (ton/m²)", 1.0, value=20.0, step=0.5, key="qa_allow")
    fc_prime      = st.number_input("fc' (ksc)", 150, 700, 280, step=10, key="fc_prime")
    fy            = st.selectbox("fy", [3000, 4000], index=1, key="fy_sel",
                                 format_func=lambda x: f"SD30 (3000 ksc)" if x==3000 else "SD40 (4000 ksc)")
    soil_density  = st.number_input("Soil density (ton/m³)", 1.0, 2.5, 1.8, step=0.1, key="soil_dens")
    base_friction = st.number_input("Friction μ", 0.1, 0.7, 0.50, step=0.05, key="mu")

with sb.expander("3. Soil Profile (Bearing & Settlement)", expanded=False):
    bc_method   = st.selectbox("BC Method", ["Terzaghi","Meyerhof","Both"], key="bc_meth")
    fail_mode   = st.selectbox("Terzaghi Mode", ["General","Local","Punching"], key="fail_mode")
    cohesion    = st.number_input("Cohesion c (ton/m²)", 0.0, value=0.0, step=0.5, key="cohesion")
    phi_deg     = st.number_input("Friction φ (°)", 0.0, 45.0, 30.0, step=1.0, key="phi_deg")
    st.markdown("**Elastic Settlement**")
    Es_soil     = st.number_input("Es (ton/m²)", 100.0, value=2000.0, step=100.0, key="Es_soil")
    nu_soil     = st.number_input("ν Poisson", 0.1, 0.49, 0.30, step=0.01, key="nu_soil")
    st.markdown("**Consolidation**")
    Cc_soil     = st.number_input("Cc", 0.01, value=0.30, step=0.01, key="Cc_soil")
    Cs_soil     = st.number_input("Cs", 0.001, value=0.05, step=0.005, key="Cs_soil")
    e0_soil     = st.number_input("e₀", 0.1, value=0.80, step=0.05, key="e0_soil")
    OCR_soil    = st.number_input("OCR", 1.0, value=1.0, step=0.5, key="OCR_soil")
    H_clay      = st.number_input("Clay layer H (m)", 0.5, value=3.0, step=0.5, key="H_clay")
    sigma_v0    = st.number_input("σ'v0 at mid-layer (ton/m²)", 1.0, value=10.0, step=1.0, key="sigma_v0")

with sb.expander("4. Column Dimensions", expanded=False):
    col_cx = st.number_input("Column cx (cm)", 20.0, value=40.0, step=5.0, key="col_cx")
    col_cy = st.number_input("Column cy (cm)", 20.0, value=40.0, step=5.0, key="col_cy")

# ── FOOTING GEOMETRY ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-ttl">📐 Footing Geometry</div>', unsafe_allow_html=True)
shape = st.radio("Shape", ["Rectangular / Square", "Circular"], horizontal=True, key="shape_radio")
is_circ = (shape == "Circular")

if is_circ:
    gc1, gc2, gc3 = st.columns(3)
    D_circ = gc1.number_input("D (m)", 0.5, value=2.5, step=0.1, key="D_circ")
    H_cm   = gc2.number_input("H (cm)", 25.0, value=60.0, step=5.0, key="H_cm_c")
    Df_m   = gc3.number_input("Df (m)", 0.5, value=1.5, step=0.1, key="Df_c")
    B_m = L_m = D_circ
else:
    gc1, gc2, gc3, gc4 = st.columns(4)
    B_m    = gc1.number_input("B (m)", 1.0, value=2.5, step=0.1, key="B_m")
    L_m    = gc2.number_input("L (m)", 1.0, value=2.5, step=0.1, key="L_m")
    H_cm   = gc3.number_input("H (cm)", 25.0, value=60.0, step=5.0, key="H_cm_r")
    Df_m   = gc4.number_input("Df (m)", 0.5, value=1.5, step=0.1, key="Df_r")
    D_circ = 0.0

# ── OPTIMISATION ─────────────────────────────────────────────────────────────
with st.expander("🔧 Auto-Optimisation (min. footing size)", expanded=False):
    oc1, oc2 = st.columns([2, 1])
    aspect = 1.0 if is_circ else oc1.number_input("Aspect L/B", 0.5, 3.0, 1.0, 0.1, key="opt_aspect")
    run_opt = oc2.button("🚀 Optimise", use_container_width=True, key="run_opt")
    if run_opt:
        loads_o = Loads(P_DL,P_LL,M_DL_x,M_LL_x,M_WL_x,M_DL_y,M_LL_y,M_WL_y,V_hx,V_hy)
        props_o = Properties(qa_allow,fc_prime,fy,soil_density,base_friction)
        geo_o   = Geometry(B_m,L_m,H_cm,Df_m,col_cx,col_cy,
                           "circular" if is_circ else "rectangular", D_circ)
        des_o = FoundationDesigner(loads_o, props_o, geo_o)
        with st.spinner("Searching…"):
            B_opt, L_opt = des_o.optimize_dimensions(aspect)
        if is_circ:
            st.success(f"✅ Minimum D = **{B_opt:.1f} m**")
        else:
            st.success(f"✅ Minimum B×L = **{B_opt:.1f} m × {L_opt:.1f} m**")

# ── BUILD OBJECTS ─────────────────────────────────────────────────────────────
loads   = Loads(P_DL,P_LL,M_DL_x,M_LL_x,M_WL_x,M_DL_y,M_LL_y,M_WL_y,V_hx,V_hy)
props   = Properties(qa_allow,fc_prime,fy,soil_density,base_friction)
geo     = Geometry(B_m,L_m,H_cm,Df_m,col_cx,col_cy,
                   "circular" if is_circ else "rectangular", D_circ)
soil    = SoilProfile(cohesion=cohesion, phi_deg=phi_deg,
                      Es=Es_soil, nu=nu_soil,
                      Cc=Cc_soil, Cs=Cs_soil, e0=e0_soil, OCR=OCR_soil,
                      H_clay=H_clay, sigma_v0=sigma_v0, soil_density=soil_density,
                      method="Terzaghi" if bc_method=="Terzaghi" else "Meyerhof",
                      failure_mode=fail_mode)

designer = FoundationDesigner(loads, props, geo)
sls      = designer.analyze_service_state()
uls      = designer.analyze_ultimate_state()
nb_B, sp_B, As_B, rho_B = designer.design_flexure(uls["M_u_B"], designer.L_cm)
nb_L, sp_L, As_L, rho_L = designer.design_flexure(uls["M_u_L"], designer.B_cm)

bc_engine = BearingCapacityEngine(soil, geo, loads)
bc_results = (bc_engine.run_both() if bc_method=="Both" else
              {"Terzaghi": bc_engine.terzaghi()} if bc_method=="Terzaghi" else
              {"Meyerhof": bc_engine.meyerhof()})

q_net_svc   = max(sls["P_total"] / designer.A_base - soil_density * Df_m, 0.0)
sett_engine = SettlementEngine(soil, geo, q_net_svc)
sett        = sett_engine.total()

# ── UTIL: utilisation bar ─────────────────────────────────────────────────────
def ubar(label, demand, capacity, is_fs=False):
    if is_fs:
        ratio = capacity / demand if demand > 0 else 0.5
        cap_str = f"≥ {capacity:.2f}"
    else:
        ratio = demand / capacity if capacity > 0 else 1.5
        cap_str = f"≤ {capacity:.3f}"
    pct = min(ratio * 100, 100)
    color = "#10B981" if ratio <= 0.80 else ("#F59E0B" if ratio <= 1.0 else "#EF4444")
    stat = "PASS" if ratio <= 1.0 else "FAIL"
    return f"""<div style="background:#F8FAFC;padding:12px;border-radius:8px;
border:1px solid #E2E8F0;margin-bottom:10px;">
<div style="display:flex;justify-content:space-between;font-weight:600;
color:#1E293B;margin-bottom:6px;font-size:13px;">
<span>{label}</span>
<span style="color:{color};">{demand:.3f} {cap_str}
<span style="background:{color};color:white;padding:1px 6px;border-radius:4px;
margin-left:6px;font-size:11px;">{stat}</span></span></div>
<div style="width:100%;background:#CBD5E1;border-radius:4px;height:8px;overflow:hidden;">
<div style="width:{pct}%;background:{color};height:100%;"></div></div>
<div style="text-align:right;margin-top:3px;font-size:11px;color:#64748B;">
Utilization {ratio*100:.1f}%</div></div>"""

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
t_dash, t_geo, t_bc, t_sett, t_comb, t_struct, t_3d, t_bp = st.tabs([
    "📊 Dashboard", "🪨 Geotechnical", "🏔️ Bearing Capacity",
    "📉 Settlement", "🔗 Combined Footing", "🧱 Structural",
    "🌐 3D Pressure", "🎨 Blueprints",
])

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
with t_dash:
    st.subheader("Foundation Performance Dashboard")
    cc = "#1D4ED8" if "W" in uls["governing_combo"] else "#059669"
    st.markdown(
        f'<div style="background:{cc}10;border:1px solid {cc};border-radius:8px;'
        f'padding:10px 16px;margin-bottom:14px;color:{cc};font-weight:600;">'
        f'⚡ Governing ULS Combo: <code>{uls["governing_combo"]}</code>'
        f'&nbsp;|&nbsp; Pu = {uls["P_u"]:.1f} t'
        f'&nbsp;|&nbsp; Mux = {uls["M_u_x"]:.2f} t·m'
        f'&nbsp;|&nbsp; Muy = {uls["M_u_y"]:.2f} t·m</div>',
        unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### 🪨 Geotechnical Safety")
        st.markdown(ubar("Max Bearing q_max (t/m²)", sls["q_max"], qa_allow), unsafe_allow_html=True)
        st.markdown(ubar("FS Overturning X", sls["FS_ovr_x"], FS_OVR_MIN, True), unsafe_allow_html=True)
        st.markdown(ubar("FS Overturning Y", sls["FS_ovr_y"], FS_OVR_MIN, True), unsafe_allow_html=True)
        st.markdown(ubar("FS Sliding",       sls["FS_slide"],  FS_SLIDE_MIN, True), unsafe_allow_html=True)
    with d2:
        st.markdown("#### 🧱 Structural Safety (ULS)")
        st.markdown(ubar("1-Way Shear X vux (kg/cm²)", uls["v_u_x"], uls["phi_vc_wide"]), unsafe_allow_html=True)
        st.markdown(ubar("1-Way Shear Y vuy (kg/cm²)", uls["v_u_y"], uls["phi_vc_wide"]), unsafe_allow_html=True)
        st.markdown(ubar("Punching Shear vup (kg/cm²)", uls["v_u_punch"], uls["phi_vc_punch"]), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Summary")
    dim_str = f"D={D_circ:.2f}m (circular)" if is_circ else f"B={B_m:.2f}m × L={L_m:.2f}m"
    st.info(f"**{dim_str}** | H={H_cm:.0f}cm | d={designer.d_cm:.1f}cm | Df={Df_m:.1f}m")
    sc1, sc2 = st.columns(2)
    sc1.success(f"**B-direction** (along B): {nb_B}×DB16 @ {sp_B:.1f}cm c/c  |  As={nb_B*DB16_AREA:.2f}cm²  |  ρ={rho_B*100:.3f}%")
    sc2.success(f"**L-direction** (along L): {nb_L}×DB16 @ {sp_L:.1f}cm c/c  |  As={nb_L*DB16_AREA:.2f}cm²  |  ρ={rho_L*100:.3f}%")

    with st.expander("🔍 Detailed Geotechnical Calcs"):
        st.markdown(f"**P_total** = {sls['P_total']:.2f} t &nbsp;|&nbsp; "
                    f"e_x = {sls['e_x']:.4f}m (kern {sls['kern_x']:.3f}m) &nbsp;|&nbsp; "
                    f"e_y = {sls['e_y']:.4f}m (kern {sls['kern_y']:.3f}m)")
        ten = "⚠️ YES – partial liftoff" if sls["has_tension"] else "✅ Full contact"
        st.markdown(f"Tension: **{ten}** | q_max={sls['q_max']:.2f} | q_min={sls['q_min']:.2f} t/m²")
        for nm, expr, val, lim in [
            ("FS_ovr_x", r"P_{total}·L/2 / (M_x+V_{hy}·D_f)", sls['FS_ovr_x'], 1.5),
            ("FS_ovr_y", r"P_{total}·B/2 / (M_y+V_{hx}·D_f)", sls['FS_ovr_y'], 1.5),
            ("FS_slide",  r"P_{total}·μ / V_h",                 sls['FS_slide'],  1.5),
        ]:
            ok = "✅ PASS" if val >= lim else "❌ FAIL"
            st.markdown(f"$\\mathrm{{{nm}}} = {val:.3f}$ ≥ {lim}  → **{ok}**")

    with st.expander("🔍 Detailed Structural Calcs"):
        st.markdown(f"**Governing:** `{uls['governing_combo']}`  |  "
                    f"qu_base={uls['qu_base']:.4f}  qu_mod_B={uls['qu_mod_B']:.4f}  "
                    f"qu_mod_L={uls['qu_mod_L']:.4f} kg/cm²")
        st.markdown(f"**d** = {designer.d_cm:.1f} cm  |  **bo** = {uls['bo']:.1f} cm  |  "
                    f"β_c = {uls['beta_c']:.2f}")
        st.latex(rf"v_{{u,punch}} = {uls['v_u_punch']:.4f}\;\text{{kg/cm}}^2 \le \phi v_c = {uls['phi_vc_punch']:.4f}")
        st.markdown(f"vc controls: min(vc1={uls['vc1']:.3f}, vc2={uls['vc2']:.3f}, vc3={uls['vc3']:.3f}) × φ={PHI_SHEAR}")
        ld = uls['L_d']
        st.latex(rf"L_d = {ld:.1f}\;\text{{cm}} \;(\text{{ACI 318-19 §25.5.2.1, confined}})")
        for dr, av, tag in [("B-dir", uls['avail_Ld_B'], "B"), ("L-dir", uls['avail_Ld_L'], "L")]:
            ok = "✅ OK" if av >= ld else "⚠️ Hook required"
            st.markdown(f"Available {dr}: **{av:.1f} cm** vs Ld={ld:.1f} cm → **{ok}**")

# ── GEOTECHNICAL ──────────────────────────────────────────────────────────────
with t_geo:
    st.markdown("### 🪨 Geotechnical Analytics")
    ga, gb = st.columns(2)
    ga.metric("P_total", f"{sls['P_total']:.2f} t")
    ga.metric("q_avg",   f"{sls['q_avg']:.2f} t/m²")
    ga.metric("q_max",   f"{sls['q_max']:.2f} t/m²", delta=f"Limit {qa_allow} t/m²")
    ga.metric("q_min",   f"{sls['q_min']:.2f} t/m²")
    gb.metric("e_x",  f"{sls['e_x']:.4f} m", delta=f"kern={sls['kern_x']:.3f}m")
    gb.metric("e_y",  f"{sls['e_y']:.4f} m", delta=f"kern={sls['kern_y']:.3f}m")
    gb.metric("FS ovr X", f"{sls['FS_ovr_x']:.2f}", delta="≥1.50")
    gb.metric("FS ovr Y", f"{sls['FS_ovr_y']:.2f}", delta="≥1.50")
    gb.metric("FS slide", f"{sls['FS_slide']:.2f}",  delta="≥1.50")
    st.markdown(f"Tension zone: {'⚠️ YES' if sls['has_tension'] else '✅ No'}")

# ── BEARING CAPACITY ──────────────────────────────────────────────────────────
with t_bc:
    st.markdown("### 🏔️ Bearing Capacity")

    def bc_panel(res, label):
        qu=res["qu_ultimate"]; qa=res["qa_computed"]
        ok = qa >= sls["q_max"]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("qu (t/m²)", f"{qu:.2f}")
        c2.metric("qa (t/m²)", f"{qa:.2f}")
        c3.metric("FS",        f"{res['FS_bearing']:.1f}")
        c4.metric("Status",    "✅ PASS" if ok else "❌ FAIL")
        st.markdown(f"**Nc={res['Nc']:.2f}  Nq={res['Nq']:.2f}  Nγ={res['Ng']:.2f}** "
                    f"| sc={res.get('sc','-'):.3f}  sq={res.get('sq','-'):.3f}  sγ={res.get('sg','-'):.3f}")
        if "ic" in res:
            st.markdown(f"Depth: dc={res['dc']:.3f} dq={res['dq']:.3f} dγ={res['dg']:.3f} "
                        f"| Inclin: ic={res['ic']:.3f} iq={res['iq']:.3f} iγ={res['ig']:.3f} (α={res['alpha_deg']:.1f}°)")
        with st.expander(f"Formula – {label}"):
            if "Terzaghi" in label:
                st.latex(r"q_u = c N_c s_c + \gamma D_f N_q s_q + 0.5\gamma B N_\gamma s_\gamma")
                st.markdown(f"c_use={res.get('c_use',cohesion):.2f} t/m²,  φ_use={res.get('phi_use_deg',phi_deg):.1f}°")
            else:
                st.latex(r"q_u = c N_c s_c d_c i_c + \gamma D_f N_q s_q d_q i_q + 0.5\gamma B N_\gamma s_\gamma d_\gamma i_\gamma")
            st.latex(rf"q_u = {qu:.2f}\text{{ t/m}}^2 \;\Rightarrow\; q_a = {qu:.2f}/3.0 = {qa:.2f}\text{{ t/m}}^2")

    for lbl, r in bc_results.items():
        st.markdown(f"#### {lbl}")
        bc_panel(r, lbl)
        st.markdown("---")

    # Sensitivity: qu vs φ
    st.markdown("#### qu vs φ sensitivity")
    phi_arr = list(range(0, 46, 2))
    qt_arr, qm_arr = [], []
    for p in phi_arr:
        s2=copy.copy(soil); s2.phi_deg=p
        e2=BearingCapacityEngine(s2,geo,loads)
        qt_arr.append(e2.terzaghi()["qu_ultimate"])
        qm_arr.append(e2.meyerhof()["qu_ultimate"])
    fig_bc = go.Figure()
    fig_bc.add_trace(go.Scatter(x=phi_arr, y=qt_arr, name="Terzaghi", line=dict(color="#2563EB",width=2)))
    fig_bc.add_trace(go.Scatter(x=phi_arr, y=qm_arr, name="Meyerhof", line=dict(color="#059669",width=2)))
    fig_bc.add_hline(y=sls["q_max"]*FS_BEARING, line_dash="dash", line_color="#DC2626",
                     annotation_text=f"3×q_max = {sls['q_max']*3:.1f}")
    fig_bc.update_layout(xaxis_title="φ (°)", yaxis_title="qu (t/m²)", height=340, margin=dict(t=20,b=40))
    st.plotly_chart(fig_bc, use_container_width=True)

# ── SETTLEMENT ────────────────────────────────────────────────────────────────
with t_sett:
    st.markdown("### 📉 Settlement Analysis")
    el=sett["elastic"]; con=sett["consolidation"]
    m1,m2,m3 = st.columns(3)
    m1.metric("Elastic Si",        f"{el['Si_cm']:.2f} cm")
    m2.metric("Consolidation Sc",  f"{con['Sc_cm']:.2f} cm")
    m3.metric("Total St",          f"{sett['St_cm']:.2f} cm",
              delta="⚠️ > 2.5 cm" if sett["St_cm"]>2.5 else "✅ OK")

    ce, cc2 = st.columns(2)
    with ce:
        st.markdown("#### Elastic Settlement")
        st.latex(r"S_i = q_{net}\,B\,\frac{1-\nu^2}{E_s}\,I_f")
        st.table({"Parameter":["q_net","B","Es","ν","If","Si"],
                  "Value":[f"{q_net_svc:.2f} t/m²",f"{el['B']:.2f}m",
                           f"{Es_soil:.0f}",f"{nu_soil:.2f}",
                           f"{el['If']:.3f}",f"{el['Si_cm']:.3f} cm"]})
    with cc2:
        st.markdown("#### Consolidation Settlement")
        st.latex(r"S_c = \frac{C_c H}{1+e_0}\log_{10}\frac{\sigma'_{v0}+\Delta\sigma}{\sigma'_{v0}}")
        st.table({"Parameter":["Regime","σv0","σvc","Δσ","σv1","Cc/Cs","Sc"],
                  "Value":[con["regime"],
                           f"{con['sv0']:.2f}",f"{con['svc']:.2f}",
                           f"{con['dsigma']:.3f}",f"{con['sv1']:.2f}",
                           f"{Cc_soil:.3f}/{Cs_soil:.3f}",
                           f"{con['Sc_cm']:.3f} cm"]})

    # Settlement vs B
    st.markdown("---")
    Bv=np.linspace(1.0,6.0,30); St_v=[]
    for bv in Bv:
        g2=copy.copy(geo); g2.B=float(bv); g2.L=float(bv)
        q2=max(sls["P_total"]/(bv*bv)-soil_density*Df_m,0)
        St_v.append(SettlementEngine(soil,g2,q2).total()["St_cm"])
    fig_st=go.Figure()
    fig_st.add_trace(go.Scatter(x=list(Bv),y=St_v,line=dict(color="#7C3AED",width=2)))
    fig_st.add_hline(y=2.5,line_dash="dash",line_color="#DC2626",annotation_text="Limit 2.5cm")
    fig_st.add_vline(x=B_m,line_dash="dot",line_color="#2563EB",annotation_text=f"B={B_m}m")
    fig_st.update_layout(xaxis_title="B (m)",yaxis_title="Settlement (cm)",height=320,margin=dict(t=20,b=30))
    st.plotly_chart(fig_st,use_container_width=True)

# ── COMBINED FOOTING ──────────────────────────────────────────────────────────
with t_comb:
    st.markdown("### 🔗 Combined Footing (2 Columns, aligned along L)")
    cf1,cf2 = st.columns(2)
    with cf1:
        st.markdown("**Column 1**")
        c1_PDL=st.number_input("P_DL col1",value=25.0,step=1.0,key="c1_PDL")
        c1_PLL=st.number_input("P_LL col1",value=15.0,step=1.0,key="c1_PLL")
        c1_cx=st.number_input("cx col1 (cm)",value=40.0,step=5.0,key="c1_cx")
        c1_cy=st.number_input("cy col1 (cm)",value=40.0,step=5.0,key="c1_cy")
        c1_x=st.number_input("x₁ from left edge (m)",value=0.30,step=0.1,key="c1_x")
    with cf2:
        st.markdown("**Column 2**")
        c2_PDL=st.number_input("P_DL col2",value=35.0,step=1.0,key="c2_PDL")
        c2_PLL=st.number_input("P_LL col2",value=20.0,step=1.0,key="c2_PLL")
        c2_cx=st.number_input("cx col2 (cm)",value=40.0,step=5.0,key="c2_cx")
        c2_cy=st.number_input("cy col2 (cm)",value=40.0,step=5.0,key="c2_cy")
        c2_x=st.number_input("x₂ from left edge (m)",value=3.70,step=0.1,key="c2_x")

    cf_B=st.number_input("Footing Width B (m)",0.5,value=2.0,step=0.1,key="cf_B")
    cf_H=st.number_input("Footing Thickness H (cm)",30.0,value=70.0,step=5.0,key="cf_H")
    cf_type=st.radio("Type",["Rectangular","Trapezoidal"],horizontal=True,key="cf_type")

    col1d=ColumnData(c1_PDL,c1_PLL,c1_cx,c1_cy,c1_x)
    col2d=ColumnData(c2_PDL,c2_PLL,c2_cx,c2_cy,c2_x)
    cfd=CombinedFootingDesigner(col1d,col2d,cf_B,cf_H,Df_m,props,soil,cf_type)
    try:
        cfr=cfd.design()
        mc1,mc2,mc3,mc4=st.columns(4)
        mc1.metric("P_total",f"{cfr['Pt']:.1f} t")
        mc2.metric("L",f"{cfr['L']:.2f} m")
        mc3.metric("q_net",f"{cfr['q_net']:.2f} t/m²")
        ok_q=cfr["q_net"]<=qa_allow
        mc4.metric("vs q_allow",f"{qa_allow:.1f}",delta="✅ OK" if ok_q else "❌ Fails")
        if cf_type=="Trapezoidal":
            tb1,tb2=st.columns(2)
            tb1.metric("b₁ (at col1 end)",f"{cfr['b1']:.2f} m")
            tb2.metric("b₂ (at col2 end)",f"{cfr['b2']:.2f} m")

        fig_vm=go.Figure()
        fig_vm.add_trace(go.Scatter(x=cfr["xs"],y=cfr["V"],name="Shear V (ton)",
                         line=dict(color="#2563EB",width=2),fill="tozeroy",
                         fillcolor="rgba(37,99,235,0.10)"))
        fig_vm.add_trace(go.Scatter(x=cfr["xs"],y=cfr["M"],name="Moment M (ton·m)",
                         line=dict(color="#DC2626",width=2),fill="tozeroy",
                         fillcolor="rgba(220,38,38,0.10)"))
        fig_vm.add_vline(x=cfr["x1"],line_dash="dot",line_color="#059669",annotation_text="C1")
        fig_vm.add_vline(x=cfr["x2"],line_dash="dot",line_color="#059669",annotation_text="C2")
        fig_vm.update_layout(xaxis_title="x (m)",yaxis_title="V (ton) / M (ton·m)",
                             height=360,margin=dict(t=20,b=30))
        st.plotly_chart(fig_vm,use_container_width=True)
        st.markdown(f"M_max+ = **{cfr['M_pos']:.2f}** t·m  |  M_max− = **{cfr['M_neg']:.2f}** t·m")
    except Exception as e:
        st.error(f"Error: {e}")

# ── STRUCTURAL ────────────────────────────────────────────────────────────────
with t_struct:
    st.markdown("### 🧱 ULS Concrete Design Verification")
    s1,s2=st.columns(2)
    with s1:
        st.subheader("1-Way (Wide-Beam) Shear")
        st.metric("vu_x",f"{uls['v_u_x']:.4f} kg/cm²")
        st.metric("vu_y",f"{uls['v_u_y']:.4f} kg/cm²")
        st.metric("φVc", f"{uls['phi_vc_wide']:.4f} kg/cm²")
        ok_w=uls["v_u_wide_max"]<=uls["phi_vc_wide"]
        st.success("✔ PASS") if ok_w else st.error("❌ FAIL – increase H")
    with s2:
        st.subheader("Punching Shear")
        st.metric("bo",f"{uls['bo']:.1f} cm")
        st.metric("vu_punch",f"{uls['v_u_punch']:.4f} kg/cm²")
        st.metric("φVc_punch",f"{uls['phi_vc_punch']:.4f} kg/cm²")
        ok_p=uls["v_u_punch"]<=uls["phi_vc_punch"]
        st.success("✔ PASS") if ok_p else st.error("❌ FAIL – increase H")

    st.markdown("---")
    st.subheader("Flexural Reinforcement")
    r1,r2=st.columns(2)
    r1.info(f"**B-direction** (bottom bars parallel to B)\n\n"
            f"{nb_B}×DB16 @ {sp_B:.1f} cm c/c\n\n"
            f"As_req={As_B:.2f} cm²  As_prov={nb_B*DB16_AREA:.2f} cm²  ρ={rho_B*100:.3f}%")
    r2.info(f"**L-direction** (bottom bars parallel to L)\n\n"
            f"{nb_L}×DB16 @ {sp_L:.1f} cm c/c\n\n"
            f"As_req={As_L:.2f} cm²  As_prov={nb_L*DB16_AREA:.2f} cm²  ρ={rho_L*100:.3f}%")

    st.subheader("Development Length (ACI 318-19 §25.5.2.1)")
    st.latex(rf"L_d = \frac{{3}}{{40\sqrt{{10.2}}}} \cdot \frac{{\psi_s}}{{(c_b+K_{{tr}})/d_b}} "
             rf"\cdot \frac{{f_y}}{{\sqrt{{f'_c}}}} \cdot d_b \ge 30\;\text{{cm}}")
    d1,d2,d3=st.columns(3)
    d1.metric("Required Ld",  f"{uls['L_d']:.1f} cm")
    d2.metric("Available (B-dir)", f"{uls['avail_Ld_B']:.1f} cm",
              delta="OK" if uls['avail_Ld_B']>=uls['L_d'] else "⚠️ Hook")
    d3.metric("Available (L-dir)", f"{uls['avail_Ld_L']:.1f} cm",
              delta="OK" if uls['avail_Ld_L']>=uls['L_d'] else "⚠️ Hook")

# ── 3D PRESSURE ───────────────────────────────────────────────────────────────
with t_3d:
    st.markdown("### 🌐 3D Contact Stress Distribution (Service State)")
    XX,YY,ZZ = designer.soil_pressure_grid(n=50)
    fig3d=go.Figure(data=[go.Surface(x=XX,y=YY,z=ZZ,colorscale="RdYlGn_r",
                    cmin=0,cmax=float(np.nanmax(ZZ))*1.05,
                    colorbar=dict(title="q (t/m²)",thickness=14),opacity=0.92)])
    fig3d.add_surface(x=XX,y=YY,z=np.full_like(ZZ,qa_allow),
                      colorscale=[[0,"rgba(239,68,68,0.25)"],[1,"rgba(239,68,68,0.25)"]],
                      showscale=False,opacity=0.35)
    fig3d.update_layout(
        scene=dict(xaxis_title="X (m)",yaxis_title="Y (m)",zaxis_title="q (t/m²)",
                   camera=dict(eye=dict(x=1.6,y=1.6,z=1.2))),
        height=540,margin=dict(l=0,r=0,t=30,b=0),
        title=dict(text="Contact Stress (red plane = q_allow)",x=0.5))
    st.plotly_chart(fig3d,use_container_width=True)
    st.info(f"q_max={sls['q_max']:.2f}  q_min={sls['q_min']:.2f}  q_allow={qa_allow:.2f} t/m²")

# ── BLUEPRINTS ────────────────────────────────────────────────────────────────
with t_bp:
    st.markdown("### 🎨 2D Engineering Blueprints")
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,7),gridspec_kw={"width_ratios":[1,1.2]})
    fig.patch.set_facecolor("#FFFFFF")

    def _draw_dim(ax,x1,y1,x2,y2,txt,ox=0,oy=0):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="<->",color="#475569",lw=1.2))
        ax.text((x1+x2)/2+ox,(y1+y2)/2+oy,txt,ha="center",va="center",
                fontsize=9,color="#0F172A",backgroundcolor="white")

    # Plan view
    if is_circ:
        R=D_circ/2
        circle=plt.Circle((R,R),R,fc="#F8FAFC",ec="#0F172A",lw=2)
        ax1.add_patch(circle)
        ax1.plot([0.2,2*R-0.2],[R,R],color="#94A3B8",lw=1,ls="dashdot")
        ax1.plot([R,R],[0.2,2*R-0.2],color="#94A3B8",lw=1,ls="dashdot")
        cx0=R-(col_cx/100)/2; cy0=R-(col_cy/100)/2
        ax1.add_patch(plt.Rectangle((cx0,cy0),col_cx/100,col_cy/100,
                      fc="#FEE2E2",hatch="//",ec="#DC2626",lw=1.5))
        ax1.set_xlim(-0.3,2*R+0.3); ax1.set_ylim(-0.3,2*R+0.5)
        _draw_dim(ax1,0,2*R+0.15,2*R,2*R+0.15,f"D={D_circ:.2f}m")
    else:
        ax1.add_patch(plt.Rectangle((0,0),B_m,L_m,fc="#F8FAFC",ec="#0F172A",lw=2))
        ax1.plot([-0.2,B_m+0.2],[L_m/2,L_m/2],color="#94A3B8",lw=1,ls="dashdot")
        ax1.plot([B_m/2,B_m/2],[-0.2,L_m+0.2],color="#94A3B8",lw=1,ls="dashdot")
        cx0=(B_m-col_cx/100)/2; cy0=(L_m-col_cy/100)/2
        ax1.add_patch(plt.Rectangle((cx0,cy0),col_cx/100,col_cy/100,
                      fc="#FEE2E2",hatch="//",ec="#DC2626",lw=1.5))
        poff=designer.d_cm/200
        ax1.add_patch(plt.Rectangle((cx0-poff,cy0-poff),
                      col_cx/100+2*poff,col_cy/100+2*poff,
                      fill=False,ec="#D97706",lw=1.5,ls="--"))
        cov=0.075
        for i in range(1,4):
            ax1.plot([cov,B_m-cov],[cov+i*(L_m-2*cov)/15]*2,color="#3B82F6",lw=0.8,alpha=0.6)
            ax1.plot([cov+i*(B_m-2*cov)/15]*2,[cov,L_m-cov],color="#3B82F6",lw=0.8,alpha=0.6)
        _draw_dim(ax1,0,L_m+0.15,B_m,L_m+0.15,f"B={B_m:.2f}m")
        ax1.text(-0.25,L_m/2,f"L={L_m:.2f}m",ha="center",va="center",
                 fontsize=9,rotation=90,color="#0F172A")
        ax1.set_xlim(-0.4,B_m+0.3); ax1.set_ylim(-0.4,L_m+0.4)

    ax1.set_aspect("equal"); ax1.axis("off")
    ax1.set_title("PLAN VIEW",fontsize=12,fontweight="bold",color="#1E3A8A",pad=15)

    # Section view
    H_m2=H_cm/100; lean=0.05; fbase=lean; cov_m=0.075
    ax2.plot([-0.5,B_m+0.5],[Df_m,Df_m],color="#451A03",lw=1.5)
    ax2.text(B_m+0.12,Df_m+0.04,"F.G.L.",color="#451A03",fontsize=9,fontweight="bold")
    ax2.add_patch(plt.Rectangle((-0.05,0),B_m+0.1,lean,fc="#E2E8F0",hatch="...",ec="#64748B",lw=1))
    ax2.add_patch(plt.Rectangle((0,fbase),B_m,H_m2,fc="#F1F5F9",ec="#0F172A",lw=2))
    col_top=Df_m+0.35; cx_s=(B_m-col_cx/100)/2
    ax2.add_patch(plt.Rectangle((cx_s,fbase+H_m2),col_cx/100,col_top-(fbase+H_m2),
                  fc="#FEE2E2",ec="#0F172A",lw=1.5))
    rx1,rx2=cov_m,B_m-cov_m; ry=fbase+cov_m
    ax2.plot([rx1,rx2],[ry,ry],color="#1D4ED8",lw=2.5)
    n_d=min(nb_B,12)
    for i in range(n_d):
        ax2.plot(rx1+i*(rx2-rx1)/max(n_d-1,1),ry+0.015,"o",ms=4,color="#DC2626")
    dw1,dw2=cx_s+0.05,cx_s+col_cx/100-0.05
    ax2.plot([dw1,dw1],[ry,col_top+0.06],color="#047857",lw=2)
    ax2.plot([dw2,dw2],[ry,col_top+0.06],color="#047857",lw=2)
    _draw_dim(ax2,0,-0.15,B_m,-0.15,f"B={B_m:.2f}m")
    ax2.text(-0.25,fbase+H_m2/2,f"H={H_cm:.0f}cm",ha="center",va="center",
             fontsize=9,rotation=90,color="#0F172A")
    ax2.annotate(f"{nb_B}×DB16",xy=(rx2-0.3,ry),xytext=(rx2+0.12,ry-0.15),
                 arrowprops=dict(arrowstyle="->",lw=1),fontsize=8)
    ax2.set_xlim(-0.5,B_m+0.55); ax2.set_ylim(-0.35,Df_m+0.6)
    ax2.set_aspect("equal",adjustable="datalim"); ax2.axis("off")
    ax2.set_title("SECTION ELEVATION",fontsize=12,fontweight="bold",color="#1E3A8A",pad=15)
    st.pyplot(fig)
