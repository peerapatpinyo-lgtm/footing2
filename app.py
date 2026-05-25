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
        self.d_cm = self.geo.H_cm - 7.5 # Effective depth
        self.A_base = self.geo.B * self.geo.L
        
    def analyze_service_state(self):
        P_service = self.loads.P_DL + self.loads.P_LL
        M_service_x = self.loads.M_DL_x + self.loads.M_LL_x
        M_service_y = self.loads.M_DL_y + self.loads.M_LL_y
        
        W_footing = self.A_base * (self.geo.H_cm / 100) * 2.4
        W_overburden = self.A_base * (self.geo.Df - (self.geo.H_cm / 100)) * self.props.soil_density
        P_total = P_service + W_footing + W_overburden
        
        e_x = M_service_y / P_total if P_total > 0 else 0
        e_y = M_service_x / P_total if P_total > 0 else 0
        
        kern_x, kern_y = self.geo.B / 6, self.geo.L / 6
        has_tension = (e_x > kern_x) or (e_y > kern_y)
        
        q_avg = P_total / self.A_base
        q_mod_x = (M_service_y * (self.geo.B / 2)) / ((self.geo.B**3 * self.geo.L) / 12)
        q_mod_y = (M_service_x * (self.geo.L / 2)) / ((self.geo.B * self.geo.L**3) / 12)
        
        if not has_tension:
            q_max = q_avg + q_mod_x + q_mod_y
            q_min = max(0.0, q_avg - q_mod_x - q_mod_y)
        else:
            B_prime = max(self.geo.B - 2 * e_x, 0.1)
            L_prime = max(self.geo.L - 2 * e_y, 0.1)
            q_max = P_total / (B_prime * L_prime) * (4.0 / 3.0 if (e_x > kern_x and e_y > kern_y) else 1.0)
            q_min = 0.0

        M_res_x = P_total * (self.geo.L / 2)
        M_ovr_x = M_service_x + (self.loads.V_hy * self.geo.Df)
        FS_ovr_x = M_res_x / M_ovr_x if M_ovr_x > 0 else float('inf')
        
        M_res_y = P_total * (self.geo.B / 2)
        M_ovr_y = M_service_y + (self.loads.V_hx * self.geo.Df)
        FS_ovr_y = M_res_y / M_ovr_y if M_ovr_y > 0 else float('inf')
        
        V_h_total = math.sqrt(self.loads.V_hx**2 + self.loads.V_hy**2)
        FS_slide = (P_total * self.props.base_friction) / V_h_total if V_h_total > 0 else float('inf')

        return {
            "P_total": P_total, "e_x": e_x, "e_y": e_y, "kern_x": kern_x, "kern_y": kern_y,
            "has_tension": has_tension, "q_max": q_max, "q_min": q_min,
            "FS_ovr_x": FS_ovr_x, "FS_ovr_y": FS_ovr_y, "FS_slide": FS_slide
        }

    def analyze_ultimate_state(self):
        P_u = max(1.4 * self.loads.P_DL, 1.2 * self.loads.P_DL + 1.6 * self.loads.P_LL)
        M_u_x = max(1.4 * self.loads.M_DL_x, 1.2 * self.loads.M_DL_x + 1.6 * self.loads.M_LL_x + 1.0 * self.loads.M_WL_x)
        M_u_y = max(1.4 * self.loads.M_DL_y, 1.2 * self.loads.M_DL_y + 1.6 * self.loads.M_LL_y + 1.0 * self.loads.M_WL_y)
        
        qu_base = (P_u * 1000) / (self.B_cm * self.L_cm)
        qu_mod_x = (M_u_y * 1e5 * (self.B_cm / 2)) / ((self.L_cm * self.B_cm**3) / 12)
        qu_mod_y = (M_u_x * 1e5 * (self.L_cm / 2)) / ((self.B_cm * self.L_cm**3) / 12)
        qu_max = qu_base + qu_mod_x + qu_mod_y
        
        crit_x = max(0.0, ((self.B_cm - self.geo.cx) / 2) - self.d_cm)
        v_u_wide_x = (qu_max * self.L_cm * crit_x) / (self.L_cm * self.d_cm)
        crit_y = max(0.0, ((self.L_cm - self.geo.cy) / 2) - self.d_cm)
        v_u_wide_y = (qu_max * self.B_cm * crit_y) / (self.B_cm * self.d_cm)
        phi_v_c_wide = 0.75 * 0.53 * math.sqrt(self.props.fc_prime)
        
        bo = 2 * ((self.geo.cx + self.d_cm) + (self.geo.cy + self.d_cm))
        area_punch = (self.geo.cx + self.d_cm) * (self.geo.cy + self.d_cm)
        v_u_punch = (qu_max * ((self.B_cm * self.L_cm) - area_punch)) / (bo * self.d_cm)
        
        beta_c = max(self.geo.cx, self.geo.cy) / min(self.geo.cx, self.geo.cy)
        vc1 = 0.27 * (2 + 4/beta_c) * math.sqrt(self.props.fc_prime)
        vc2 = 0.27 * ((40 * self.d_cm / bo) + 2) * math.sqrt(self.props.fc_prime)
        vc3 = 1.06 * math.sqrt(self.props.fc_prime)
        phi_v_c_punch = 0.75 * min(vc1, vc2, vc3)

        cant_x = (self.B_cm - self.geo.cx) / 2
        M_ux = (qu_max * self.L_cm * (cant_x ** 2)) / 2
        cant_y = (self.L_cm - self.geo.cy) / 2
        M_uy = (qu_max * self.B_cm * (cant_y ** 2)) / 2
        
        # Anchorage Length
        db_size = 16 
        L_d = (self.props.fy / (1.4 * math.sqrt(self.props.fc_prime))) * (db_size / 10)
        available_L_d = min(cant_x, cant_y) - 7.5

        return {
            "v_u_wide_x": v_u_wide_x, "v_u_wide_y": v_u_wide_y, "v_u_wide_max": max(v_u_wide_x, v_u_wide_y),
            "phi_v_c_wide": phi_v_c_wide, "v_u_punch": v_u_punch, "phi_v_c_punch": phi_v_c_punch,
            "M_ux": M_ux, "M_uy": M_uy, "bo": bo, "L_d": L_d, "available_L_d": available_L_d
        }

    def design_flexure(self, M_u, width_cm):
        rho_min = 0.0018 if self.props.fy >= 4000 else 0.0020
        R_n = M_u / (0.90 * width_cm * (self.d_cm ** 2))
        m = self.props.fy / (0.85 * self.props.fc_prime)
        
        discriminant = 1 - (2 * m * R_n / self.props.fy)
        rho_req = (1 / m) * (1 - math.sqrt(max(0, discriminant))) if discriminant > 0 else rho_min
        
        As_req = max(rho_req, rho_min) * width_cm * self.d_cm
        bars_req = max(5, math.ceil(As_req / 2.01)) # DB16
        spacing = (width_cm - 15) / (bars_req - 1)
        return bars_req, spacing

# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="Advanced Biaxial Foundation Suite", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main-header { font-size:30px; font-weight:700; color:#0F172A; margin-bottom:5px; }
    .sub-header { font-size:16px; color:#475569; margin-bottom:30px; }
    .section-title { font-size:20px; font-weight:600; color:#1E3A8A; border-left: 6px solid #2563EB; padding-left: 12px; margin-top:25px; margin-bottom:15px; }
    .metric-card { background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1); }
    .status-pass { color: #059669; font-weight: 700; }
    .status-fail { color: #DC2626; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏗️ Advanced Biaxial Foundation Engineering Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ultimate Limit State (ULS) & Serviceability Design for Eccentric Footings | ACI 318 & International Codes Compliant</div>', unsafe_allow_html=True)

# --- SIDEBAR INPUTS ---
st.sidebar.header("📥 1. Structural Service Loads")
with st.sidebar.expander("📊 Axial, Moments & Shears", expanded=True):
    P_DL = st.sidebar.number_input("Dead Load: P_DL (tons)", min_value=0.0, value=30.0, step=1.0)
    P_LL = st.sidebar.number_input("Live Load: P_LL (tons)", min_value=0.0, value=18.0, step=1.0)
    st.sidebar.markdown("**Bending Moments**")
    M_DL_x = st.sidebar.number_input("M_DL x (ton-m)", value=3.5, step=0.5)
    M_LL_x = st.sidebar.number_input("M_LL x (ton-m)", value=2.0, step=0.5)
    M_WL_x = st.sidebar.number_input("M_WL x (Wind) (ton-m)", value=1.5, step=0.5)
    M_DL_y = st.sidebar.number_input("M_DL y (ton-m)", value=2.5, step=0.5)
    M_LL_y = st.sidebar.number_input("M_LL y (ton-m)", value=1.5, step=0.5)
    M_WL_y = st.sidebar.number_input("M_WL y (Wind) (ton-m)", value=1.0, step=0.5)
    st.sidebar.markdown("**Base Shears**")
    V_hx = st.sidebar.number_input("Horizontal Shear X: V_hx (tons)", value=2.0, step=0.1)
    V_hy = st.sidebar.number_input("Horizontal Shear Y: V_hy (tons)", value=1.8, step=0.1)

st.sidebar.header("🧱 2. Material & Geotechnical Specs")
with st.sidebar.expander("Properties", expanded=False):
    qa_allow = st.sidebar.number_input("Allowable Soil Bearing: q_allow (tsf or ton/m²)", min_value=1.0, value=20.0, step=0.5)
    fc_prime = st.sidebar.number_input("Concrete Compressive Strength: fc' (ksc)", min_value=150, value=280, step=10)
    fy = st.sidebar.selectbox("Rebar Yield Strength (fy)", [3000, 4000], index=1, format_func=lambda x: f"Grade 40 (fy={x})" if x==3000 else f"Grade 60 / SD40 (fy={x})")
    soil_density = st.sidebar.number_input("Surcharge Soil Density (ton/m³)", value=1.8, step=0.1)
    base_friction = st.sidebar.number_input("Base Friction Coefficient (μ)", min_value=0.1, max_value=0.7, value=0.50, step=0.05)

st.sidebar.header("📐 3. Column Dimensions")
col_bx = st.sidebar.number_input("Column Width X: cx (cm)", value=40.0, step=5.0)
col_by = st.sidebar.number_input("Column Depth Y: cy (cm)", value=40.0, step=5.0)

# --- MAIN GEOMETRY INTERACTION ---
st.markdown('<div class="section-title">📐 Footing Geometry Optimization</div>', unsafe_allow_html=True)
gc1, gc2, gc3, gc4 = st.columns(4)
B_m = gc1.number_input("Footing Width X: B (m)", min_value=1.0, value=2.5, step=0.1)
L_m = gc2.number_input("Footing Length Y: L (m)", min_value=1.0, value=2.5, step=0.1)
H_cm = gc3.number_input("Total Thickness: H (cm)", min_value=25.0, value=60.0, step=5.0)
Df_m = gc4.number_input("Embedment Depth: Df (m)", min_value=0.5, value=1.5, step=0.1)

# --- EXECUTE ENGINEERING LOGIC ---
loads = Loads(P_DL, P_LL, M_DL_x, M_LL_x, M_WL_x, M_DL_y, M_LL_y, M_WL_y, V_hx, V_hy)
props = Properties(qa_allow, fc_prime, fy, soil_density, base_friction)
geo = Geometry(B_m, L_m, H_cm, Df_m, col_bx, col_by)

designer = FoundationDesigner(loads, props, geo)
sls = designer.analyze_service_state()
uls = designer.analyze_ultimate_state()
bars_count_x, space_x = designer.design_flexure(uls["M_ux"], designer.L_cm)
bars_count_y, space_y = designer.design_flexure(uls["M_uy"], designer.B_cm)

# --- ENTERPRISE PRESENTATION TABS ---
tab_dash, tab_geo, tab_struct, tab_draw = st.tabs([
    "📊 Safety Performance Dashboard", 
    "🪨 Geotechnical Analytics", 
    "🧱 Structural Concrete Design", 
    "🎨 2D Engineering Blueprints"
])

with tab_dash:
    st.subheader("💡 Foundation Key Performance Indicators (KPIs)")
    mc1, mc2, mc3, mc4 = st.columns(4)
    
    with mc1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Max Soil Pressure ($q_{max}$)", f"{sls['q_max']:.2f} t/m²", f"Limit: {qa_allow:.1f}")
        status = "<span class='status-pass'>PASS</span>" if sls['q_max'] <= qa_allow else "<span class='status-fail'>FAIL</span>"
        st.markdown(f"Bearing Capacity: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with mc2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("FS Overturning (X)", f"{sls['FS_ovr_x']:.2f}", "Target ≥ 1.50")
        status = "<span class='status-pass'>PASS</span>" if sls['FS_ovr_x'] >= 1.5 else "<span class='status-fail'>FAIL</span>"
        st.markdown(f"Stability Status: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with mc3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("FS Sliding", f"{sls['FS_slide']:.2f}", "Target ≥ 1.50")
        status = "<span class='status-pass'>PASS</span>" if sls['FS_slide'] >= 1.5 else "<span class='status-fail'>FAIL</span>"
        st.markdown(f"Sliding Status: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with mc4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Two-Way Shear Stress", f"{uls['v_u_punch']:.1f} ksc", f"Limit: {uls['phi_v_c_punch']:.1f}")
        status = "<span class='status-pass'>PASS</span>" if uls['v_u_punch'] <= uls['phi_v_c_punch'] else "<span class='status-fail'>FAIL</span>"
        st.markdown(f"Punching Status: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Executive Construction Procurement Specification")
    st.info(f"📐 **Footing Dimensions:** Width B = {B_m:.2f} m | Length L = {L_m:.2f} m | Thickness H = {H_cm:.0f} cm (Effective Depth d = {designer.d_cm:.1f} cm)")
    
    sc1, sc2 = st.columns(2)
    sc1.success(f"🧲 **X-Axis Bottom Mesh:** Provide **{bars_count_x}x DB16** Bars spaced at **@{space_x:.1f} cm** c/c")
    sc2.success(f"🧲 **Y-Axis Bottom Mesh:** Provide **{bars_count_y}x DB16** Bars spaced at **@{space_y:.1f} cm** c/c")

with tab_geo:
    st.markdown("### 🪨 Advanced Soil Geotechnical Analytics")
    st.write("#### 1. Real-time Contact Biaxial Stress Distribution")
    st.write(f"- Total Vertical Load acting on soil matrix ($P_{{total}}$): `{sls['P_total']:.2f}` tons")
    st.write(f"- Calculated Eccentricities: $e_x$ = `{sls['e_x']:.3f}` m, $e_y$ = `{sls['e_y']:.3f}` m")
    st.write(f"- Safe Kern Boundaries Area: $B/6$ = `{sls['kern_x']:.3f}` m, $L/6$ = `{sls['kern_y']:.3f}` m")
    
    if not sls['has_tension']:
        st.success(f"✔️ Resultant falls inside Kern boundary. Base stays in full contact compression ($q_{{min}}$ = `{sls['q_min']:.2f}` t/m²)")
    else:
        st.warning(f"⚠️ Resultant falls outside Kern area! Partial liftoff / soil tension occurs. True $q_{{max}}$ was resolved via contact area optimization.")
        
    st.write("#### 2. Global Safety Factors")
    st.write(f"- Factor of Safety against Overturning (X-Axis): `{sls['FS_ovr_x']:.2f}` (Required $\ge 1.50$)")
    st.write(f"- Factor of Safety against Overturning (Y-Axis): `{sls['FS_ovr_y']:.2f}` (Required $\ge 1.50$)")
    st.write(f"- Factor of Safety against Base Sliding Failure: `{sls['FS_slide']:.2f}` (Required $\ge 1.50$)")

with tab_struct:
    st.markdown("### 🧱 Ultimate Limit State (ULS) Concrete Verification")
    
    st.write("#### 1. Broad One-Way (Wide-Beam) Structural Shear")
    st.write(f"- Shear Demand on X-Axis ($v_{{ux}}$) = `{uls['v_u_wide_x']:.2f}` kg/cm²")
    st.write(f"- Shear Demand on Y-Axis ($v_{{uy}}$) = `{uls['v_u_wide_y']:.2f}` kg/cm²")
    st.write(f"- Design Ultimate Shear Capacity ($\phi v_c$) = `{uls['phi_v_c_wide']:.2f}` kg/cm²")
    if uls['v_u_wide_max'] <= uls['phi_v_c_wide']:
        st.success("✔️ PASS: Footing cross-section provides sufficient concrete shear resistance.")
    else:
        st.error("❌ FAIL: Excessive wide-beam shear. Increase thickness (H) immediately.")

    st.write("#### 2. Two-Way Column Punching Validation")
    st.write(f"- Critical perimeter length at $d/2$ from column face ($b_0$) = `{uls['bo']:.1f}` cm")
    st.write(f"- Actual Punching Shear Demand ($v_u$) = `{uls['v_u_punch']:.2f}` kg/cm²")
    st.write(f"- Governing Code Punching Limit ($\phi v_c$) = `{uls['phi_v_c_punch']:.2f}` kg/cm²")
    if uls['v_u_punch'] <= uls['phi_v_c_punch']:
        st.success("✔️ PASS: Concrete punch geometry is highly stable and secure.")
    else:
        st.error("❌ FAIL: Concrete punching shear failure imminent! Increase footing depth (H).")

    st.write("#### 3. Steel Development Anchorage Length Validation")
    st.write(f"- Required tension anchorage length ($L_d$) = `{uls['L_d']:.1f}` cm")
    st.write(f"- Available projection distance inside footing ($L_{{available}}$) = `{uls['available_L_d']:.1f}` cm")
    if uls['available_L_d'] >= uls['L_d']:
        st.success("✔️ PASS: Straight bars are perfectly anchored.")
    else:
        st.warning(f"⚠️ WARNING: Insufficient anchorage length. Standard 90-degree hooks required at rebar terminations.")

with tab_draw:
    st.markdown("### 🎨 2D AutoCAD-Style Engineering Blueprints")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    
    # DRAWING 1: PLAN VIEW
    ax1.add_patch(plt.Rectangle((0, 0), B_m, L_m, color='#F8FAFC', ec='#1E3A8A', lw=2.5, label='Footing Boundary'))
    cx = (B_m - (col_bx/100)) / 2
    cy = (L_m - (col_by/100)) / 2
    ax1.add_patch(plt.Rectangle((cx, cy), col_bx/100, col_by/100, color='#FEE2E2', ec='#DC2626', lw=2, label='Column Core'))
    
    p_off = (designer.d_cm / 100) / 2
    ax1.add_patch(plt.Rectangle((cx - p_off, cy - p_off), (col_bx/100) + 2*p_off, (col_by/100) + 2*p_off, fill=False, ec='#D97706', lw=1.5, ls='--', label='Punching Surface (d/2)'))
    
    # Bottom reinforcement mesh matrix lines
    for i in range(min(bars_count_x, 12)):
        pos = 0.075 + i * ((L_m - 0.15) / (min(bars_count_x, 12) - 1))
        ax1.plot([0.075, B_m - 0.075], [pos, pos], color='#3B82F6', lw=1.2, alpha=0.8)
    for i in range(min(bars_count_y, 12)):
        pos = 0.075 + i * ((B_m - 0.15) / (min(bars_count_y, 12) - 1))
        ax1.plot([pos, pos], [0.075, L_m - 0.075], color='#1D4ED8', lw=1.2, alpha=0.8)

    ax1.set_xlim(-0.3, B_m + 0.3)
    ax1.set_ylim(-0.3, L_m + 0.3)
    ax1.set_aspect('equal')
    ax1.set_title("REINFORCEMENT DETAILED MESH (PLAN VIEW)", fontsize=11, fontweight='bold', color='#1E3A8A')
    ax1.axis('off')
    ax1.text(B_m/2, L_m + 0.05, f"L = {L_m:.2f} m", ha='center', va='bottom', weight='bold')
    ax1.text(B_m + 0.05, L_m/2, f"B = {B_m:.2f} m", ha='left', va='center', weight='bold', rotation=-90)
    ax1.legend(loc='lower left', fontsize=8)

    # DRAWING 2: CROSS SECTION VIEW
    ax2.plot([-0.4, B_m + 0.4], [Df_m, Df_m], color='#78350F', lw=2, label='Finished Ground Level')
    ax2.add_patch(plt.Rectangle((-0.04, 0.04), B_m + 0.08, 0.04, color='#CBD5E1', ec='#64748B', lw=1)) # Lean concrete
    ax2.add_patch(plt.Rectangle((0, 0.08), B_m, H_cm/100, color='#E2E8F0', ec='#1E3A8A', lw=2.5)) # Footing block
    ax2.add_patch(plt.Rectangle((cx, 0.08 + H_cm/100), col_bx/100, Df_m - (0.08 + H_cm/100) + 0.3, color='#FEE2E2', ec='#DC2626', lw=2)) # Column stem
    
    # Bottom mesh with standard 90 deg hooks
    ry = 0.08 + 0.075
    ax2.plot([0.075, B_m - 0.075], [ry, ry], color='#3B82F6', lw=2.5)
    ax2.plot([0.075, 0.075], [ry, ry + 0.15], color='#3B82F6', lw=2.5)
    ax2.plot([B_m - 0.075, B_m - 0.075], [ry, ry + 0.15], color='#3B82F6', lw=2.5)
    
    # Main column starter dowel rebars
    ax2.plot([cx + 0.04, cx + 0.04], [0.08 + 0.075, Df_m + 0.2], color='#DC2626', lw=2)
    ax2.plot([cx + (col_bx/100) - 0.04, cx + (col_bx/100) - 0.04], [0.08 + 0.075, Df_m + 0.2], color='#DC2626', lw=2)

    ax2.set_xlim(-0.4, B_m + 0.4)
    ax2.set_ylim(-0.1, Df_m + 0.5)
    ax2.set_title("STRUCTURAL EMBEDMENT ELEVATION (SECTION VIEW)", fontsize=11, fontweight='bold', color='#1E3A8A')
    ax2.axis('off')
    ax2.text(B_m + 0.03, 0.08 + (H_cm/200), f"H = {H_cm:.0f} cm", va='center', weight='bold')
    ax2.text(-0.05, Df_m, f"Df = {Df_m:.2f} m", ha='right', va='center', color='#78350F', weight='bold')

    st.pyplot(fig)
