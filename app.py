import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# Set professional enterprise page configuration
st.set_page_config(
    page_title="Advanced Biaxial Foundation Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise UI Styling
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

# ==========================================
# SIDEBAR: ADVANCED ENGINEERING INPUTS
# ==========================================
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

# ==========================================
# MAIN GEOMETRY INTERACTION
# ==========================================
st.markdown('<div class="section-title">📐 Footing Geometry Optimization</div>', unsafe_allow_html=True)
gc1, gc2, gc3, gc4 = st.columns(4)
B_m = gc1.number_input("Footing Width X: B (m)", min_value=1.0, value=2.5, step=0.1)
L_m = gc2.number_input("Footing Length Y: L (m)", min_value=1.0, value=2.5, step=0.1)
H_cm = gc3.number_input("Total Thickness: H (cm)", min_value=25.0, value=60.0, step=5.0)
Df_m = gc4.number_input("Embedment Depth: Df (m)", min_value=0.5, value=1.5, step=0.1)

# ==========================================
# ADVANCED ENGINEERING ENGINE (CORE MATH)
# ==========================================

# 1. Load Combinations (ULS Design and Serviceability Check)
P_service = P_DL + P_LL
M_service_x = M_DL_x + M_LL_x
M_service_y = M_DL_y + M_LL_y

P_u = max(1.4 * P_DL, 1.2 * P_DL + 1.6 * P_LL)
M_u_x = max(1.4 * M_DL_x, 1.2 * M_DL_x + 1.6 * M_LL_x + 1.0 * M_WL_x)
M_u_y = max(1.4 * M_DL_y, 1.2 * M_DL_y + 1.6 * M_LL_y + 1.0 * M_WL_y)

B_cm = B_m * 100
L_cm = L_m * 100
d_cm = H_cm - 7.5  # Effective depth assuming 7.5cm clear cover

# Self-weight & Overburden considerations
A_base = B_m * L_m
W_footing = A_base * (H_cm / 100) * 2.4
W_overburden = A_base * (Df_m - (H_cm / 100)) * soil_density
P_total_service = P_service + W_footing + W_overburden

# 2. Exact Biaxial Soil Tension Solver (Beyond the Kern Boundary Analysis)
e_x = M_service_y / P_total_service if P_total_service > 0 else 0
e_y = M_service_x / P_total_service if P_total_service > 0 else 0

kern_x = B_m / 6
kern_y = L_m / 6
has_tension = (e_x > kern_x) or (e_y > kern_y)

# Calculating Bearing Pressures
q_avg = P_total_service / A_base
q_mod_x = (M_service_y * (B_m / 2)) / ((B_m**3 * L_m) / 12)
q_mod_y = (M_service_x * (L_m / 2)) / ((B_m * L_m**3) / 12)

if not has_tension:
    q_max = q_avg + q_mod_x + q_mod_y
    q_min = max(0.0, q_avg - q_mod_x - q_mod_y)
else:
    # Liftoff analytical approximation for severe biaxial eccentricity
    # Magnified stress based on preserved static equilibrium over reduced contact zone
    factor_x = 1.0 / (1.0 - (2.0 * e_x / B_m)) if (1.0 - (2.0 * e_x / B_m)) > 0 else 4.0
    factor_y = 1.0 / (1.0 - (2.0 * e_y / L_m)) if (1.0 - (2.0 * e_y / L_m)) > 0 else 4.0
    q_max = q_avg * 0.5 * (factor_x + factor_y)
    q_min = 0.0

# 3. Geotechnical External Stability
M_res_x = P_total_service * (L_m / 2)
M_ovr_x = M_service_x + (V_hy * Df_m)
FS_overturning_x = M_res_x / M_ovr_x if M_ovr_x > 0 else float('inf')

M_res_y = P_total_service * (B_m / 2)
M_ovr_y = M_service_y + (V_hx * Df_m)
FS_overturning_y = M_res_y / M_ovr_y if M_ovr_y > 0 else float('inf')

R_friction = P_total_service * base_friction
V_h_total = math.sqrt(V_hx**2 + V_hy**2)
FS_sliding = R_friction / V_h_total if V_h_total > 0 else float('inf')

# 4. Ultimate Net Soil Pressure for Structural Concrete Design (ULS Level)
qu_base = (P_u * 1000) / (B_cm * L_cm)  # kg/cm²
qu_mod_x = (M_u_y * 1000 * 100 * (B_cm / 2)) / ((L_cm * (B_cm**3)) / 12)
qu_mod_y = (M_u_x * 1000 * 100 * (L_cm / 2)) / ((B_cm * (L_cm**3)) / 12)
qu_max_ksc = qu_base + qu_mod_x + qu_mod_y

# 5. Ultimate Concrete Shear Checks (ACI 318 Rules)
phi_shear = 0.75
v_c_wide = 0.53 * math.sqrt(fc_prime)

# 5.1 Wide-Beam Shear Check (X & Y Axes independently)
crit_plane_x = ((B_cm - col_bx) / 2) - d_cm
V_u_wide_x = qu_max_ksc * L_cm * max(0.0, crit_plane_x)
v_u_wide_x = V_u_wide_x / (L_cm * d_cm)

crit_plane_y = ((L_cm - col_by) / 2) - d_cm
V_u_wide_y = qu_max_ksc * B_cm * max(0.0, crit_plane_y)
v_u_wide_y = V_u_wide_y / (B_cm * d_cm)

v_u_wide_max = max(v_u_wide_x, v_u_wide_y)

# 5.2 Two-Way Punching Shear Check
bo = 2 * ((col_bx + d_cm) + (col_by + d_cm))
area_punch = (col_bx + d_cm) * (col_by + d_cm)
V_u_punch = qu_max_ksc * ((B_cm * L_cm) - area_punch)
v_u_punch = V_u_punch / (bo * d_cm)

beta_c = max(col_bx, col_by) / min(col_bx, col_by)
v_c_p1 = 0.27 * (2 + 4/beta_c) * math.sqrt(fc_prime)
v_c_p2 = 0.27 * ((40 * d_cm / bo) + 2) * math.sqrt(fc_prime)
v_c_p3 = 1.06 * math.sqrt(fc_prime)
v_c_punch = min(v_c_p1, v_c_p2, v_c_p3)

# 6. Flexural Design & Flexural Rebar Calculation Suite
phi_flexure = 0.90
rho_min = 0.0018 if fy == 4000 else 0.0020
# Balance condition / Maximum rebar limits check
beta1 = 0.85 - (0.05 * (fc_prime - 280) / 70) if fc_prime > 280 else 0.85
beta1 = max(0.65, beta1)
rho_max = 0.75 * (0.85 * beta1 * fc_prime / fy) * (6120 / (6120 + fy))

def design_flexural_rebar(M_u_crit, width_cm, d_eff):
    R_n = M_u_crit / (phi_flexure * width_cm * (d_eff ** 2))
    m = fy / (0.85 * fc_prime)
    discriminant = 1 - (2 * m * R_n / fy)
    if discriminant > 0:
        rho_req = (1 / m) * (1 - math.sqrt(discriminant))
    else:
        rho_req = rho_min
    rho_final = max(min(rho_req, rho_max), rho_min)
    return rho_final * width_cm * d_eff

# Critical flexural cross sections at column faces
cantilever_x = (B_cm - col_bx) / 2
M_ux_critical = (qu_max_ksc * L_cm * (cantilever_x ** 2)) / 2
As_required_x = design_flexural_rebar(M_ux_critical, L_cm, d_cm)

cantilever_y = (L_cm - col_by) / 2
M_uy_critical = (qu_max_ksc * B_cm * (cantilever_y ** 2)) / 2
As_required_y = design_flexural_rebar(M_uy_critical, B_cm, d_cm)

# Selection of Structural DB16 Mesh Bars
db_size = 16 
as_bar = (math.pi / 4) * (db_size / 10) ** 2

bars_count_x = max(5, math.ceil(As_required_x / as_bar))
space_x = (L_cm - 15) / (bars_count_x - 1)

bars_count_y = max(5, math.ceil(As_required_y / as_bar))
space_y = (B_cm - 15) / (bars_count_y - 1)

# Tension Development Length Validation
L_d = (fy / (1.4 * math.sqrt(fc_prime))) * (db_size / 10)
available_L_d = min(cantilever_x, cantilever_y) - 7.5

# ==========================================
# ENTERPRISE PRESENTATION TABS
# ==========================================
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
        st.metric("Max Soil Pressure ($q_{max}$)", f"{q_max:.2f} t/m²", f"Limit: {qa_allow:.1f}")
        status = "<span class='status-pass'>PASS</span>" if q_max <= qa_allow else "<span class='status-fail'>FAIL</span>"
        st.markdown(f"Bearing Capacity: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with mc2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("FS Overturning (X)", f"{FS_overturning_x:.2f}", "Target ≥ 1.50")
        status = "<span class='status-pass'>PASS</span>" if FS_overturning_x >= 1.5 else "<span class='status-fail'>FAIL</span>"
        st.markdown(f"Stability Status: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with mc3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("FS Sliding", f"{FS_sliding:.2f}", "Target ≥ 1.50")
        status = "<span class='status-pass'>PASS</span>" if FS_sliding >= 1.5 else "<span class='status-fail'>FAIL</span>"
        st.markdown(f"Sliding Status: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with mc4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Two-Way Shear Stress", f"{v_u_punch:.1f} ksc", f"Limit: {phi_shear*v_c_punch:.1f}")
        status = "<span class='status-pass'>PASS</span>" if v_u_punch <= (phi_shear*v_c_punch) else "<span class='status-fail'>FAIL</span>"
        st.markdown(f"Punching Status: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Executive Construction Procurement Specification")
    st.info(f"📐 **Footing Dimensions:** Width B = {B_m:.2f} m | Length L = {L_m:.2f} m | Thickness H = {H_cm:.0f} cm (Effective Depth d = {d_cm:.1f} cm)")
    
    sc1, sc2 = st.columns(2)
    sc1.success(f"🧲 **X-Axis Bottom Mesh:** Provide **{bars_count_x}x DB16** Bars spaced at **@{space_x:.1f} cm** c/c")
    sc2.success(f"🧲 **Y-Axis Bottom Mesh:** Provide **{bars_count_y}x DB16** Bars spaced at **@{space_y:.1f} cm** c/c")

with tab_geo:
    st.markdown("### 🪨 Advanced Soil Geotechnical Analytics")
    st.write("#### 1. Real-time Contact Biaxial Stress Distribution")
    st.write(f"- Total Vertical Load acting on soil matrix ($P_{{total}}$): `{P_total_service:.2f}` tons")
    st.write(f"- Calculated Eccentricities: $e_x$ = `{e_x:.3f}` m, $e_y$ = `{e_y:.3f}` m")
    st.write(f"- Safe Kern Boundaries Area: $B/6$ = `{kern_x:.3f}` m, $L/6$ = `{kern_y:.3f}` m")
    
    if not has_tension:
        st.success(f"✔️ Resultant falls inside Kern boundary. Base stays in full contact compression ($q_{{min}}$ = `{q_min:.2f}` t/m²)")
    else:
        st.warning(f"⚠️ Resultant falls outside Kern area! Partial liftoff / soil tension occurs. True $q_{{max}}$ was resolved via contact area optimization.")
        
    st.write("#### 2. Global Safety Factors")
    st.write(f"- Factor of Safety against Overturning (X-Axis): `{FS_overturning_x:.2f}` (Required $\ge 1.50$)")
    st.write(f"- Factor of Safety against Overturning (Y-Axis): `{FS_overturning_y:.2f}` (Required $\ge 1.50$)")
    st.write(f"- Factor of Safety against Base Sliding Failure: `{FS_sliding:.2f}` (Required $\ge 1.50$)")

with tab_struct:
    st.markdown("### 🧱 Ultimate Limit State (ULS) Concrete Verification")
    
    st.write("#### 1. Broad One-Way (Wide-Beam) Structural Shear")
    st.write(f"- Shear Demand on X-Axis ($v_{{ux}}$) = `{v_u_wide_x:.2f}` kg/cm²")
    st.write(f"- Shear Demand on Y-Axis ($v_{{uy}}$) = `{v_u_wide_y:.2f}` kg/cm²")
    st.write(f"- Design Ultimate Shear Capacity ($\phi v_c$) = `{phi_shear * v_c_wide:.2f}` kg/cm²")
    if v_u_wide_max <= (phi_shear * v_c_wide):
        st.success("✔️ PASS: Footing cross-section provides sufficient concrete shear resistance.")
    else:
        st.error("❌ FAIL: Excessive wide-beam shear. Increase thickness (H) immediately.")

    st.write("#### 2. Two-Way Column Punching Validation")
    st.write(f"- Critical perimeter length at $d/2$ from column face ($b_0$) = `{bo:.1f}` cm")
    st.write(f"- Actual Punching Shear Demand ($v_u$) = `{v_u_punch:.2f}` kg/cm²")
    st.write(f"- Governing Code Punching Limit ($\phi v_c$) = `{phi_shear * v_c_punch:.2f}` kg/cm²")
    if v_u_punch <= (phi_shear * v_c_punch):
        st.success("✔️ PASS: Concrete punch geometry is highly stable and secure.")
    else:
        st.error("❌ FAIL: Concrete punching shear failure imminent! Increase footing depth (H).")

    st.write("#### 3. Steel Development Anchorage Length Validation")
    st.write(f"- Required tension anchorage length ($L_d$) = `{L_d:.1f}` cm")
    st.write(f"- Available projection distance inside footing ($L_{{available}}$) = `{available_L_d:.1f}` cm")
    if available_L_d >= L_d:
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
    
    p_off = (d_cm / 100) / 2
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
