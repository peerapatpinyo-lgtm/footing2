import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass

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

# ==========================================
# 2. ENGINEERING CALCULATION ENGINE (OOP)
# ==========================================
class FoundationDesigner:
    def __init__(self, loads: Loads, props: Properties, geo: Geometry):
        self.loads = loads
        self.props = props
        self.geo = geo
        
        # Derived Geometry
        self.B_cm = self.geo.B * 100
        self.L_cm = self.geo.L * 100
        self.d_cm = self.geo.H_cm - 7.5 # Effective depth (7.5cm cover)
        self.A_base = self.geo.B * self.geo.L
        
    def analyze_service_state(self):
        """Analyzes Geotechnical Serviceability Limit States (SLS)"""
        P_service = self.loads.P_DL + self.loads.P_LL
        M_service_x = self.loads.M_DL_x + self.loads.M_LL_x
        M_service_y = self.loads.M_DL_y + self.loads.M_LL_y
        
        W_footing = self.A_base * (self.geo.H_cm / 100) * 2.4
        W_overburden = self.A_base * (self.geo.Df - (self.geo.H_cm / 100)) * self.props.soil_density
        P_total = P_service + W_footing + W_overburden
        
        ex = M_service_y / P_total if P_total > 0 else 0
        ey = M_service_x / P_total if P_total > 0 else 0
        
        kern_x, kern_y = self.geo.B / 6, self.geo.L / 6
        has_tension = (ex > kern_x) or (ey > kern_y)
        
        # Rigorous Soil Pressure Calculation
        q_avg = P_total / self.A_base
        if not has_tension:
            q_mod_x = (P_total * ex * (self.geo.B / 2)) / ((self.geo.L * self.geo.B**3) / 12)
            q_mod_y = (P_total * ey * (self.geo.L / 2)) / ((self.geo.B * self.geo.L**3) / 12)
            q_max = q_avg + q_mod_x + q_mod_y
            q_min = max(0.0, q_avg - q_mod_x - q_mod_y)
        else:
            # 1-Way Liftoff Analytical Solution for dominant eccentricity
            if ex > kern_x and ey <= kern_y:
                q_max = (4 * P_total) / (3 * self.geo.L * (self.geo.B - 2 * ex))
            elif ey > kern_y and ex <= kern_x:
                q_max = (4 * P_total) / (3 * self.geo.B * (self.geo.L - 2 * ey))
            else:
                # Biaxial liftoff approximation (Conservative)
                q_max = P_total / ((self.geo.B - 2*ex) * (self.geo.L - 2*ey))
            q_min = 0.0

        # Stability
        M_res_x = P_total * (self.geo.L / 2)
        M_ovr_x = M_service_x + (self.loads.V_hy * self.geo.Df)
        FS_ovr_x = M_res_x / M_ovr_x if M_ovr_x > 0 else float('inf')
        
        M_res_y = P_total * (self.geo.B / 2)
        M_ovr_y = M_service_y + (self.loads.V_hx * self.geo.Df)
        FS_ovr_y = M_res_y / M_ovr_y if M_ovr_y > 0 else float('inf')
        
        V_h_total = math.sqrt(self.loads.V_hx**2 + self.loads.V_hy**2)
        FS_slide = (P_total * self.props.base_friction) / V_h_total if V_h_total > 0 else float('inf')

        return {
            "P_total": P_total, "ex": ex, "ey": ey, "kern_x": kern_x, "kern_y": kern_y,
            "has_tension": has_tension, "q_max": q_max, "q_min": q_min,
            "FS_ovr_x": FS_ovr_x, "FS_ovr_y": FS_ovr_y, "FS_slide": FS_slide
        }

    def analyze_ultimate_state(self):
        """Analyzes Structural Ultimate Limit States (ULS - ACI 318)"""
        P_u = max(1.4 * self.loads.P_DL, 1.2 * self.loads.P_DL + 1.6 * self.loads.P_LL)
        M_u_x = max(1.4 * self.loads.M_DL_x, 1.2 * self.loads.M_DL_x + 1.6 * self.loads.M_LL_x + 1.0 * self.loads.M_WL_x)
        M_u_y = max(1.4 * self.loads.M_DL_y, 1.2 * self.loads.M_DL_y + 1.6 * self.loads.M_LL_y + 1.0 * self.loads.M_WL_y)
        
        qu_base = (P_u * 1000) / (self.B_cm * self.L_cm)
        qu_mod_x = (M_u_y * 1e5 * (self.B_cm / 2)) / ((self.L_cm * self.B_cm**3) / 12)
        qu_mod_y = (M_u_x * 1e5 * (self.L_cm / 2)) / ((self.B_cm * self.L_cm**3) / 12)
        qu_max = qu_base + qu_mod_x + qu_mod_y
        
        # Wide Beam Shear (at distance d)
        crit_x = max(0.0, ((self.B_cm - self.geo.cx) / 2) - self.d_cm)
        v_u_wide_x = (qu_max * self.L_cm * crit_x) / (self.L_cm * self.d_cm)
        
        crit_y = max(0.0, ((self.L_cm - self.geo.cy) / 2) - self.d_cm)
        v_u_wide_y = (qu_max * self.B_cm * crit_y) / (self.B_cm * self.d_cm)
        
        phi_v_c_wide = 0.75 * 0.53 * math.sqrt(self.props.fc_prime)
        
        # Punching Shear (at distance d/2)
        bo = 2 * ((self.geo.cx + self.d_cm) + (self.geo.cy + self.d_cm))
        area_punch = (self.geo.cx + self.d_cm) * (self.geo.cy + self.d_cm)
        v_u_punch = (qu_max * ((self.B_cm * self.L_cm) - area_punch)) / (bo * self.d_cm)
        
        beta_c = max(self.geo.cx, self.geo.cy) / min(self.geo.cx, self.geo.cy)
        vc1 = 0.27 * (2 + 4/beta_c) * math.sqrt(self.props.fc_prime)
        vc2 = 0.27 * ((40 * self.d_cm / bo) + 2) * math.sqrt(self.props.fc_prime)
        vc3 = 1.06 * math.sqrt(self.props.fc_prime)
        phi_v_c_punch = 0.75 * min(vc1, vc2, vc3)

        # Flexure (at column face)
        cant_x = (self.B_cm - self.geo.cx) / 2
        M_ux = (qu_max * self.L_cm * (cant_x ** 2)) / 2
        cant_y = (self.L_cm - self.geo.cy) / 2
        M_uy = (qu_max * self.B_cm * (cant_y ** 2)) / 2

        return {
            "v_u_wide_max": max(v_u_wide_x, v_u_wide_y), "phi_v_c_wide": phi_v_c_wide,
            "v_u_punch": v_u_punch, "phi_v_c_punch": phi_v_c_punch,
            "M_ux": M_ux, "M_uy": M_uy, "cant_x": cant_x, "cant_y": cant_y
        }

    def design_flexure(self, M_u, width_cm):
        """Calculates reinforcement area ensuring ACI ductility requirements"""
        rho_min = 0.0018 if self.props.fy >= 4000 else 0.0020
        R_n = M_u / (0.90 * width_cm * (self.d_cm ** 2))
        m = self.props.fy / (0.85 * self.props.fc_prime)
        
        discriminant = 1 - (2 * m * R_n / self.props.fy)
        rho_req = (1 / m) * (1 - math.sqrt(max(0, discriminant))) if discriminant > 0 else rho_min
        
        As_req = max(rho_req, rho_min) * width_cm * self.d_cm
        bars_req = max(5, math.ceil(As_req / 2.01)) # Using DB16 area = 2.01 cm2
        spacing = (width_cm - 15) / (bars_req - 1)
        return bars_req, spacing

# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="Pro Foundation Designer", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size:32px; font-weight:800; color:#0F172A; }
    .sub-header { font-size:16px; color:#475569; margin-bottom:20px; }
    .section-title { font-size:20px; font-weight:700; color:#1E3A8A; border-left: 6px solid #3B82F6; padding-left: 10px; margin: 20px 0 10px 0; }
    .card { background-color: #F8FAFC; padding: 20px; border-radius: 8px; border: 1px solid #CBD5E1; }
    .pass { color: #059669; font-weight: bold; }
    .fail { color: #DC2626; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏗️ Pro Foundation Designer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">OOP-Architected Structural Suite for Biaxial Eccentric Footings (ACI 318)</div>', unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Loading Conditions")
    P_DL = st.number_input("P_DL (ton)", value=30.0)
    P_LL = st.number_input("P_LL (ton)", value=18.0)
    st.markdown("Moment X (ton-m)"); M_DL_x = st.number_input("MDL_x", value=3.5); M_LL_x = st.number_input("MLL_x", value=2.0); M_WL_x = st.number_input("MWL_x", value=1.5)
    st.markdown("Moment Y (ton-m)"); M_DL_y = st.number_input("MDL_y", value=2.5); M_LL_y = st.number_input("MLL_y", value=1.5); M_WL_y = st.number_input("MWL_y", value=1.0)
    st.markdown("Shear (ton)"); V_hx = st.number_input("V_hx", value=2.0); V_hy = st.number_input("V_hy", value=1.8)
    
    st.header("2. Material Specs")
    qa_allow = st.number_input("Soil q_allow (t/m2)", value=20.0)
    fc_prime = st.number_input("Concrete fc' (ksc)", value=280)
    fy = st.selectbox("Rebar fy (ksc)", [3000, 4000], index=1)
    
    st.header("3. Column Size")
    cx = st.number_input("cx (cm)", value=40.0)
    cy = st.number_input("cy (cm)", value=40.0)

# --- Main Page Geometry Inputs ---
st.markdown('<div class="section-title">📐 Geometry Optimization</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
B = col1.number_input("Width B (m)", value=2.5, step=0.1)
L = col2.number_input("Length L (m)", value=2.5, step=0.1)
H_cm = col3.number_input("Thickness H (cm)", value=60.0, step=5.0)
Df = col4.number_input("Depth Df (m)", value=1.5, step=0.1)

# --- Execute Engineering Logic ---
loads = Loads(P_DL, P_LL, M_DL_x, M_LL_x, M_WL_x, M_DL_y, M_LL_y, M_WL_y, V_hx, V_hy)
props = Properties(qa_allow, fc_prime, fy, 1.8, 0.5)
geo = Geometry(B, L, H_cm, Df, cx, cy)

designer = FoundationDesigner(loads, props, geo)
sls = designer.analyze_service_state()
uls = designer.analyze_ultimate_state()
bars_x, space_x = designer.design_flexure(uls["M_ux"], designer.L_cm)
bars_y, space_y = designer.design_flexure(uls["M_uy"], designer.B_cm)

# --- Results Presentation ---
tab1, tab2 = st.tabs(["📊 Performance KPIs & Design", "🎨 Detail Drawings"])

with tab1:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Soil Max Pressure", f"{sls['q_max']:.2f} t/m²", f"Limit: {qa_allow}")
        st.markdown(f"Status: <span class='{'pass' if sls['q_max'] <= qa_allow else 'fail'}'>{'PASS' if sls['q_max'] <= qa_allow else 'FAIL'}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("FS Overturning (Min)", f"{min(sls['FS_ovr_x'], sls['FS_ovr_y']):.2f}", "Target: ≥1.5")
        st.markdown(f"Status: <span class='{'pass' if min(sls['FS_ovr_x'], sls['FS_ovr_y']) >= 1.5 else 'fail'}'>{'PASS' if min(sls['FS_ovr_x'], sls['FS_ovr_y']) >= 1.5 else 'FAIL'}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Wide-Beam Shear", f"{uls['v_u_wide_max']:.2f} ksc", f"Limit: {uls['phi_v_c_wide']:.2f}")
        st.markdown(f"Status: <span class='{'pass' if uls['v_u_wide_max'] <= uls['phi_v_c_wide'] else 'fail'}'>{'PASS' if uls['v_u_wide_max'] <= uls['phi_v_c_wide'] else 'FAIL'}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Punching Shear", f"{uls['v_u_punch']:.2f} ksc", f"Limit: {uls['phi_v_c_punch']:.2f}")
        st.markdown(f"Status: <span class='{'pass' if uls['v_u_punch'] <= uls['phi_v_c_punch'] else 'fail'}'>{'PASS' if uls['v_u_punch'] <= uls['phi_v_c_punch'] else 'FAIL'}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.info("### 🏗️ Reinforcement Specification")
    c1, c2 = st.columns(2)
    c1.success(f"**Bottom X-Axis:** {bars_x} - DB16 @ {space_x:.1f} cm")
    c2.success(f"**Bottom Y-Axis:** {bars_y} - DB16 @ {space_y:.1f} cm")
    
    if sls['has_tension']:
        st.warning("⚠️ **Notice:** Resultant force is outside the Kern limit. Base is experiencing partial liftoff (Tension). The soil pressure shown is calculated using 1-Way Elastic Liftoff Theory.")

with tab2:
    st.markdown("### 🎨 Visual Layout")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Rectangle((0, 0), B, L, color='#F8FAFC', ec='#1E3A8A', lw=2))
    col_x, col_y = (B - cx/100)/2, (L - cy/100)/2
    ax.add_patch(plt.Rectangle((col_x, col_y), cx/100, cy/100, color='#FEE2E2', ec='#DC2626', lw=2))
    ax.set_xlim(-0.5, B + 0.5); ax.set_ylim(-0.5, L + 0.5)
    ax.set_title("Footing Plan View", fontweight='bold')
    ax.axis('off')
    st.pyplot(fig)
