import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
import plotly.graph_objects as go
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
# 3. STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="Advanced Biaxial Foundation Suite v2", layout="wide", initial_sidebar_state="expanded")

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

st.markdown('<div class="main-header">🏗️ Advanced Biaxial Foundation Engineering Suite <span style="font-size:16px;color:#64748B;">v2</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ULS & SLS Design · ACI 318-19 · Rectangular & Circular Footings · Full Wind Load Combinations · Auto-Optimization</div>', unsafe_allow_html=True)

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
with st.sidebar.expander("Properties", expanded=False):
    qa_allow   = st.sidebar.number_input("Allowable Bearing q_allow (ton/m²)", min_value=1.0, value=20.0, step=0.5)
    fc_prime   = st.sidebar.number_input("Concrete fc' (ksc)", min_value=150, value=280, step=10)
    fy         = st.sidebar.selectbox("Rebar fy", [3000, 4000], index=1,
                     format_func=lambda x: f"Grade 40 (fy={x} ksc)" if x == 3000 else f"SD40 (fy={x} ksc)")
    soil_density  = st.sidebar.number_input("Soil Density (ton/m³)", value=1.8, step=0.1)
    base_friction = st.sidebar.number_input("Base Friction μ", min_value=0.1, max_value=0.7, value=0.50, step=0.05)

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
designer = FoundationDesigner(loads, props, geo)
sls      = designer.analyze_service_state()
uls      = designer.analyze_ultimate_state()
bars_x, space_x = designer.design_flexure(uls["M_ux"], designer.L_cm)
bars_y, space_y = designer.design_flexure(uls["M_uy"], designer.B_cm)

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
tab_dash, tab_geo, tab_struct, tab_3d, tab_draw = st.tabs([
    "📊 Safety Dashboard",
    "🪨 Geotechnical",
    "🧱 Structural Design",
    "🌐 3D Soil Pressure",
    "🎨 Engineering Blueprints",
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
# TAB 3: STRUCTURAL
# ════════════════════════════════════════════
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
