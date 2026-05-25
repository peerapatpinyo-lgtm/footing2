import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
import plotly.graph_objects as go

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
    st.markdown('### 🚀 Executive Performance & Material Dashboard')
    
    # ==========================================
    # 1. PLOTLY INTERACTIVE GAUGE CHARTS (KPIs)
    # ==========================================
    def create_gauge(val, limit, title, is_fs=False):
        # ถ้าเป็น Factor of Safety (is_fs=True) ค่ายิ่งมากยิ่งดี (สีเขียวอยู่ขวา)
        # ถ้าเป็น Stress (is_fs=False) ค่ายิ่งน้อยยิ่งดี (สีเขียวอยู่ซ้าย)
        
        if is_fs:
            bar_color = "#10B981" if val >= limit else "#EF4444"
            steps = [
                {'range': [0, 1.0], 'color': "#FEE2E2"},
                {'range': [1.0, limit], 'color': "#FEF3C7"},
                {'range': [limit, max(val, limit*2)], 'color': "#D1FAE5"}
            ]
            max_val = max(val, limit * 2)
        else:
            bar_color = "#10B981" if val <= limit else "#EF4444"
            steps = [
                {'range': [0, limit*0.8], 'color': "#D1FAE5"},
                {'range': [limit*0.8, limit], 'color': "#FEF3C7"},
                {'range': [limit, max(val, limit*1.5)], 'color': "#FEE2E2"}
            ]
            max_val = max(val, limit * 1.5)

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 14, 'color': '#1E293B'}},
            delta = {'reference': limit, 'increasing': {'color': "#EF4444" if not is_fs else "#10B981"}, 'decreasing': {'color': "#10B981" if not is_fs else "#EF4444"}},
            gauge = {
                'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': bar_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': steps,
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': limit
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#0F172A"})
        return fig

    # จัดเรียง 4 เกจวัดหลัก
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1: st.plotly_chart(create_gauge(sls['q_max'], qa_allow, "Soil Pressure<br>(t/m²)"), use_container_width=True)
    with kpi2: st.plotly_chart(create_gauge(sls['FS_ovr_x'], 1.5, "Overturning FS<br>(X-Axis)", is_fs=True), use_container_width=True)
    with kpi3: st.plotly_chart(create_gauge(sls['FS_slide'], 1.5, "Sliding FS<br>(Global)", is_fs=True), use_container_width=True)
    with kpi4: st.plotly_chart(create_gauge(uls['v_u_punch'], uls['phi_v_c_punch'], "Punching Shear<br>(ksc)"), use_container_width=True)

    st.markdown("---")
    
    # ==========================================
    # 2. 3D BIAXIAL PRESSURE MAPPING & MTO
    # ==========================================
    col_3d, col_mto = st.columns([1.5, 1])
    
    with col_3d:
        st.markdown('<div class="section-title">🌐 3D Biaxial Soil Stress Distribution</div>', unsafe_allow_html=True)
        # สร้าง 3D Surface Plot แสดงการกระจายตัวของแรงดันดิน
        x_vals = np.linspace(-B_m/2, B_m/2, 20)
        y_vals = np.linspace(-L_m/2, L_m/2, 20)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # คำนวณ Pressure (Simplified linear distribution for visualization)
        P_A = sls['P_total'] / (B_m * L_m)
        Mx_Iy = (loads.M_DL_x + loads.M_LL_x) / ((B_m * L_m**3)/12)
        My_Ix = (loads.M_DL_y + loads.M_LL_y) / ((L_m * B_m**3)/12)
        
        Z = P_A + (Mx_Iy * Y) + (My_Ix * X)
        Z = np.maximum(Z, 0) # ดินรับแรงดึงไม่ได้ (Tension liftoff)

        fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Width (X)',
                yaxis_title='Length (Y)',
                zaxis_title='Pressure (t/m²)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=350
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_mto:
        st.markdown('<div class="section-title">💰 Material Take-Off (MTO)</div>', unsafe_allow_html=True)
        
        # คำนวณปริมาณวัสดุ
        vol_concrete = B_m * L_m * (H_cm/100)
        vol_lean = (B_m + 0.1) * (L_m + 0.1) * 0.05
        
        # DB16 weight = 1.58 kg/m
        len_x_bar = (B_m - 0.15) + 0.3 # รวมระยะงอขอ
        len_y_bar = (L_m - 0.15) + 0.3
        weight_steel_x = bars_count_x * len_x_bar * 1.58
        weight_steel_y = bars_count_y * len_y_bar * 1.58
        total_steel = weight_steel_x + weight_steel_y
        
        ratio_steel_conc = total_steel / vol_concrete if vol_concrete > 0 else 0

        # แสดงผลแบบกล่องการเงิน
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); padding: 20px; border-radius: 12px; color: white;">
            <h4 style="margin-top: 0; color: #94A3B8; font-size: 14px;">ESTIMATED QUANTITIES</h4>
            
            <div style="margin-bottom: 15px;">
                <span style="font-size: 13px; color: #CBD5E1;">Structural Concrete (fc' {fc_prime})</span><br>
                <span style="font-size: 24px; font-weight: 700; color: #38BDF8;">{vol_concrete:.2f} m³</span>
            </div>
            
            <div style="margin-bottom: 15px;">
                <span style="font-size: 13px; color: #CBD5E1;">Lean Concrete (5cm)</span><br>
                <span style="font-size: 20px; font-weight: 700; color: #94A3B8;">{vol_lean:.2f} m³</span>
            </div>
            
            <div style="margin-bottom: 15px;">
                <span style="font-size: 13px; color: #CBD5E1;">Reinforcement Steel (DB16)</span><br>
                <span style="font-size: 24px; font-weight: 700; color: #F87171;">{total_steel:.1f} kg</span>
            </div>
            
            <div style="border-top: 1px solid #334155; padding-top: 10px; margin-top: 10px;">
                <span style="font-size: 13px; color: #94A3B8;">Steel Ratio: </span>
                <span style="font-size: 14px; font-weight: 600; color: #10B981;">{ratio_steel_conc:.1f} kg/m³</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
    st.markdown("### 🎨 2D Professional Engineering Blueprints")
    st.markdown("แบบขยายรายละเอียดฐานรากและเหล็กเสริม (สำหรับนำไปเขียนแบบก่อสร้าง)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.patch.set_facecolor('#FFFFFF')
    
    # ==============================================================
    # DRAWING 1: PLAN VIEW (แบบแปลน)
    # ==============================================================
    # 1.1 Footing Boundary
    ax1.add_patch(plt.Rectangle((0, 0), B_m, L_m, facecolor='#F8FAFC', edgecolor='#0F172A', lw=2))
    
    # 1.2 Centerlines
    cx_pos, cy_pos = B_m / 2, L_m / 2
    ax1.plot([-0.2, B_m + 0.2], [cy_pos, cy_pos], color='#94A3B8', lw=1, linestyle='dashdot')
    ax1.plot([cx_pos, cx_pos], [-0.2, L_m + 0.2], color='#94A3B8', lw=1, linestyle='dashdot')
    
    # 1.3 Column Core
    col_x, col_y = (B_m - (col_bx/100)) / 2, (L_m - (col_by/100)) / 2
    ax1.add_patch(plt.Rectangle((col_x, col_y), col_bx/100, col_by/100, facecolor='#FEE2E2', hatch='//', edgecolor='#DC2626', lw=1.5, label='Column Core'))
    
    # 1.4 Punching Shear Perimeter (d/2)
    p_off = (designer.d_cm / 100) / 2
    ax1.add_patch(plt.Rectangle((col_x - p_off, col_y - p_off), (col_bx/100) + 2*p_off, (col_by/100) + 2*p_off, 
                                fill=False, edgecolor='#D97706', lw=1.5, linestyle='--', label='Punching Perimeter (d/2)'))
    
    # 1.5 Rebar Mesh (Showing only a few representative bars for clarity)
    cover = 0.075
    ax1.plot([cover, B_m - cover], [cover, cover], color='#2563EB', lw=2, label='Bottom Mesh X-Y')
    ax1.plot([cover, cover], [cover, L_m - cover], color='#2563EB', lw=2)
    for i in range(1, 4): # Show sample spacing
        ax1.plot([cover, B_m - cover], [cover + i*(L_m - 2*cover)/15, cover + i*(L_m - 2*cover)/15], color='#3B82F6', lw=1, alpha=0.6)
        ax1.plot([cover + i*(B_m - 2*cover)/15, cover + i*(B_m - 2*cover)/15], [cover, L_m - cover], color='#3B82F6', lw=1, alpha=0.6)

    # 1.6 Dimension Lines (CAD Style)
    def draw_dim(ax, x1, y1, x2, y2, text, offset_x=0, offset_y=0):
        ax.plot([x1, x2], [y1, y2], color='#475569', lw=1)
        ax.plot([x1, x1], [y1-0.05, y1+0.05], color='#475569', lw=1.5) # Tick
        ax.plot([x2, x2], [y2-0.05, y2+0.05], color='#475569', lw=1.5) # Tick
        ax.text((x1+x2)/2 + offset_x, (y1+y2)/2 + offset_y, text, ha='center', va='center', color='#0F172A', fontsize=9, backgroundcolor='white')

    draw_dim(ax1, 0, L_m + 0.15, B_m, L_m + 0.15, f"B = {B_m:.2f} m", offset_y=0.05) # Width
    
    ax1.plot([-0.15, -0.15], [0, L_m], color='#475569', lw=1)
    ax1.plot([-0.2, -0.1], [0, 0], color='#475569', lw=1.5)
    ax1.plot([-0.2, -0.1], [L_m, L_m], color='#475569', lw=1.5)
    ax1.text(-0.25, L_m/2, f"L = {L_m:.2f} m", ha='center', va='center', color='#0F172A', fontsize=9, rotation=90)

    ax1.set_xlim(-0.4, B_m + 0.3)
    ax1.set_ylim(-0.4, L_m + 0.4)
    ax1.set_aspect('equal')
    ax1.set_title("PLAN VIEW", fontsize=12, fontweight='bold', color='#1E3A8A', pad=15)
    ax1.axis('off')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)

    # ==============================================================
    # DRAWING 2: CROSS SECTION ELEVATION (รูปตัด)
    # ==============================================================
    lean_thick = 0.05
    footing_base_y = lean_thick
    
    # 2.1 Soil & Ground Level
    ax2.plot([-0.5, B_m + 0.5], [Df_m, Df_m], color='#451A03', lw=1.5) # FGL
    ax2.text(B_m + 0.1, Df_m + 0.05, "F.G.L.", color='#451A03', fontsize=9, fontweight='bold')
    
    # 2.2 Lean Concrete (Hatch dots)
    ax2.add_patch(plt.Rectangle((-0.05, 0), B_m + 0.1, lean_thick, facecolor='#E2E8F0', hatch='...', edgecolor='#64748B', lw=1))
    ax2.text(B_m + 0.15, lean_thick/2, "Lean Concrete 5cm", color='#64748B', fontsize=8, va='center')
    
    # 2.3 Main Footing (Hatch lines)
    H_m = H_cm / 100
    ax2.add_patch(plt.Rectangle((0, footing_base_y), B_m, H_m, facecolor='#F1F5F9', hatch='', edgecolor='#0F172A', lw=2))
    
    # 2.4 Column Stem (with breakline)
    col_top = Df_m + 0.4
    ax2.add_patch(plt.Rectangle((col_x, footing_base_y + H_m), col_bx/100, col_top - (footing_base_y + H_m), facecolor='#FEE2E2', edgecolor='#0F172A', lw=1.5))
    ax2.plot([col_x - 0.05, col_x + col_bx/100 + 0.05], [col_top, col_top + 0.05], color='#0F172A', lw=1.5) # Breakline part 1
    ax2.plot([col_x - 0.05, col_x + col_bx/100 + 0.05], [col_top + 0.05, col_top + 0.1], color='#0F172A', lw=1.5) # Breakline part 2
    
    # 2.5 Reinforcement Details
    rx1, rx2 = cover, B_m - cover
    ry = footing_base_y + cover
    
    # Bottom Rebar (X-Axis) Line with hooks
    ax2.plot([rx1, rx2], [ry, ry], color='#1D4ED8', lw=2.5)
    ax2.plot([rx1, rx1], [ry, ry + 0.15], color='#1D4ED8', lw=2.5) # 90-deg hook left
    ax2.plot([rx2, rx2], [ry, ry + 0.15], color='#1D4ED8', lw=2.5) # 90-deg hook right
    
    # Bottom Rebar (Y-Axis) Dots (Cross section)
    for i in range(min(bars_count_y, 15)):
        dot_x = rx1 + i * ((rx2 - rx1) / (min(bars_count_y, 15) - 1))
        ax2.plot(dot_x, ry + 0.02, marker='o', markersize=4, color='#DC2626')
        
    # Column Dowels
    dowel_x1 = col_x + 0.05
    dowel_x2 = col_x + (col_bx/100) - 0.05
    ax2.plot([dowel_x1, dowel_x1], [ry + 0.02, col_top + 0.1], color='#047857', lw=2)
    ax2.plot([dowel_x2, dowel_x2], [ry + 0.02, col_top + 0.1], color='#047857', lw=2)
    
    # Dowel hooks at bottom
    ax2.plot([dowel_x1, dowel_x1 - 0.1], [ry + 0.02, ry + 0.02], color='#047857', lw=2)
    ax2.plot([dowel_x2, dowel_x2 + 0.1], [ry + 0.02, ry + 0.02], color='#047857', lw=2)
    
    # 2.6 Section Dimension Lines
    draw_dim(ax2, 0, -0.15, B_m, -0.15, f"B = {B_m:.2f} m", offset_y=-0.05) # Width
    
    # Vertical Dims
    ax2.plot([-0.2, -0.1], [footing_base_y, footing_base_y], color='#475569', lw=1)
    ax2.plot([-0.2, -0.1], [footing_base_y + H_m, footing_base_y + H_m], color='#475569', lw=1)
    ax2.plot([-0.15, -0.15], [footing_base_y, footing_base_y + H_m], color='#475569', lw=1)
    ax2.text(-0.25, footing_base_y + H_m/2, f"H = {H_cm:.0f} cm", ha='center', va='center', color='#0F172A', fontsize=9, rotation=90)
    
    ax2.plot([-0.35, -0.25], [footing_base_y + H_m, footing_base_y + H_m], color='#475569', lw=1)
    ax2.plot([-0.35, -0.25], [Df_m, Df_m], color='#475569', lw=1)
    ax2.plot([-0.3, -0.3], [footing_base_y + H_m, Df_m], color='#475569', lw=1)
    ax2.text(-0.4, (footing_base_y + H_m + Df_m)/2, f"Overburden", ha='center', va='center', color='#0F172A', fontsize=8, rotation=90)
    
    ax2.set_xlim(-0.6, B_m + 0.6)
    ax2.set_ylim(-0.3, Df_m + 0.6)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.set_title("SECTION ELEVATION", fontsize=12, fontweight='bold', color='#1E3A8A', pad=15)
    ax2.axis('off')
    
    # 2.7 Annotation Callouts
    ax2.annotate(f"{bars_count_x} - DB16 Main Bar", xy=(rx2 - 0.2, ry), xytext=(rx2 + 0.2, ry - 0.2),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1), fontsize=8)
    ax2.annotate(f"{bars_count_y} - DB16 Cross Bar", xy=(rx1 + 0.2, ry + 0.02), xytext=(rx1 - 0.4, ry - 0.2),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1), fontsize=8)

    st.pyplot(fig)
