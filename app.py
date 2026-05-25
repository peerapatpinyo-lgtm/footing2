import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# ตั้งค่าคอนฟิกูเรชันหน้าเว็บแบบกว้างพิเศษระดับ Enterprise
st.set_page_config(
    page_title="Enterprise Foundation Engineering Suite (EIT & DPT Compliant)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ตกแต่ง UI ด้วย Custom CSS (แก้ไขคำสั่งแสดงผล HTML เรียบร้อย)
st.markdown("""
    <style>
    .main-header { font-size:28px; font-weight:bold; color:#1E3A8A; margin-bottom:5px; }
    .sub-header { font-size:15px; color:#4B5563; margin-bottom:25px; }
    .section-title { font-size:18px; font-weight:bold; color:#1F2937; border-left: 5px solid #1E3A8A; padding-left: 10px; margin-top:20px; margin-bottom:15px; }
    .metric-card { background-color: #F8FAFC; padding: 15px; border-radius: 8px; border: 1px solid #E2E8F0; }
    .status-pass { color: #10B981; font-weight: bold; }
    .status-fail { color: #EF4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"></div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">🏗️ Enterprise Foundation Engineering Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">โปรแกรมออกแบบฐานรากแผ่เยื้องศูนย์สองแกนระดับวิชาชีพ คลอบคลุมมาตรฐาน วสท. 1007/1008 และ มยผ. 1301/1302</div>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR: ENGINEERING INPUT PARAMETERS
# ==========================================
st.sidebar.header("📥 1. น้ำหนักบรรทุกใช้งาน (Service Loads)")
with st.sidebar.expander("📊 Axial, Moments & Shears", expanded=True):
    P_DL = st.sidebar.number_input("น้ำหนักบรรทุกคงที่ Dead Load: P_DL (ตัน)", min_value=0.0, value=25.0, step=1.0)
    P_LL = st.sidebar.number_input("น้ำหนักบรรทุกจร Live Load: P_LL (ตัน)", min_value=0.0, value=15.0, step=1.0)
    
    st.sidebar.markdown("**Bending Moments (ทิศทางแกน X และ Y)**")
    M_DL_x = st.sidebar.number_input("M_DL x (ตัน-เมตร)", value=2.0, step=0.5)
    M_LL_x = st.sidebar.number_input("M_LL x (ตัน-เมตร)", value=1.5, step=0.5)
    M_WL_x = st.sidebar.number_input("M_WL x (แรงลม) (ตัน-เมตร)", value=1.0, step=0.5)
    
    M_DL_y = st.sidebar.number_input("M_DL y (ตัน-เมตร)", value=1.5, step=0.5)
    M_LL_y = st.sidebar.number_input("M_LL y (ตัน-เมตร)", value=1.0, step=0.5)
    M_WL_y = st.sidebar.number_input("M_WL y (แรงลม) (ตัน-เมตร)", value=0.8, step=0.5)

    st.sidebar.markdown("**Horizontal Shear Forces**")
    V_hx = st.sidebar.number_input("แรงเฉือนราบทิศทาง X: V_hx (ตัน)", value=1.5, step=0.1)
    V_hy = st.sidebar.number_input("แรงเฉือนราบทิศทาง Y: V_hy (ตัน)", value=1.2, step=0.1)

st.sidebar.header("🧱 2. คุณสมบัติวัสดุ & ธรณีเทคนิค")
with st.sidebar.expander("สเปกคอนกรีต เหล็ก และชั้นดิน", expanded=False):
    qa_tsm = st.sidebar.number_input("กำลังรับน้ำหนักปลอดภัยของดิน: q_allow (ตัน/ม²)", min_value=1.0, value=16.0, step=0.5)
    fc_prime = st.sidebar.number_input("กำลังอัดประลัยทรงกระบอกคอนกรีต: fc' (ksc)", min_value=150, value=240, step=10)
    fy = st.sidebar.selectbox("ชั้นคุณภาพเหล็กเสริมหลัก (fy)", [3000, 4000], index=1, format_func=lambda x: f"SD30 (fy={x})" if x==3000 else f"SD40 (fy={x})")
    soil_density = st.sidebar.number_input("ความหนาแน่นของดินถมเหนือฐาน (ตัน/ม³)", value=1.8, step=0.1)
    base_friction = st.sidebar.number_input("สัมประสิทธิ์แรงเสียดทานใต้ท้องฐาน (μ)", min_value=0.1, max_value=0.7, value=0.45, step=0.05)

st.sidebar.header("📐 3. ขนาดหน้าตัดเสาตอม่อ")
col_bx = st.sidebar.number_input("ความกว้างเสาตอม่อด้าน X (ซม.)", value=30.0, step=5.0)
col_by = st.sidebar.number_input("ความกว้างเสาตอม่อด้าน Y (ซม.)", value=30.0, step=5.0)

# ==========================================
# INTERACTIVE GEOMETRY OPTIMIZATION (MAIN PAGE)
# ==========================================
st.markdown('<div class="section-title">📐 การกำหนดมิติเรขาคณิตของฐานราก (Footing Geometry Optimization)</div>', unsafe_allow_html=True)
gc1, gc2, gc3, gc4 = st.columns(4)
B_m = gc1.number_input("ความกว้างฐานรากทิศ X: B (เมตร)", min_value=1.0, value=2.2, step=0.1)
L_m = gc2.number_input("ความยาวฐานรากทิศ Y: L (เมตร)", min_value=1.0, value=2.2, step=0.1)
H_cm = gc3.number_input("ความหนาทั้งหมดของฐานราก: H (ซม.)", min_value=25.0, value=50.0, step=5.0)
Df_m = gc4.number_input("ความลึกจากระดับดินเดิม: Df (เมตร)", min_value=0.5, value=1.5, step=0.1)

# ==========================================
# ADVANCED ENGINEERING CALCULATION ENGINE
# ==========================================

# 1. การรวมน้ำหนักบรรทุกตามมาตรฐานประลัย (Load Combinations - SDM)
P_service = P_DL + P_LL
M_service_x = M_DL_x + M_LL_x
M_service_y = M_DL_y + M_LL_y

# ตัวคูณเพิ่มกำลัง (Factored Loads) คิดกรณีวิกฤตที่สุดระหว่าง 1.4DL หรือ 1.2DL + 1.6LL + 1.0WL
P_u = max(1.4 * P_DL, 1.2 * P_DL + 1.6 * P_LL)
M_u_x = max(1.4 * M_DL_x, 1.2 * M_DL_x + 1.6 * M_LL_x + 1.0 * M_WL_x)
M_u_y = max(1.4 * M_DL_y, 1.2 * M_DL_y + 1.6 * M_LL_y + 1.0 * M_WL_y)

# แปลงหน่วยมิติต่างๆ สู่หน่วยเซนติเมตรและกิโลกรัมสำหรับงานโครงสร้าง
B_cm = B_m * 100
L_cm = L_m * 100
d_cm = H_cm - 7.5 # ระยะความหนาประสิทธิผล หักระยะหุ้มสากล 7.5 ซม.

# คำนวณน้ำหนักตัวฐานรากและดินถมกลับ (เพื่อเช็กแรงดันดินและเสถียรภาพ)
A_base = B_m * L_m
W_footing = A_base * (H_cm / 100) * 2.4
W_soil = A_base * (Df_m - (H_cm / 100)) * soil_density
P_total_service = P_service + W_footing + W_soil

# 2. คำนวณการกระจายแรงดันดินเยื้องศูนย์สองทิศทาง (Geotechnical Stress - Service Level)
I_x = (B_m * (L_m ** 3)) / 12
I_y = ((B_m ** 3) * L_m) / 12
e_x = M_service_y / P_total_service if P_total_service > 0 else 0
e_y = M_service_x / P_total_service if P_total_service > 0 else 0

# ตรวจสอบขอบเขตพิกัด Kern (ความเค้นต้องไม่เป็นแรงดึงใต้ฐาน)
kern_x = B_m / 6
kern_y = L_m / 6
is_within_kern = (e_x <= kern_x) and (e_y <= kern_y)

q_avg = P_total_service / A_base
q_mod_x = (M_service_y * (B_m / 2)) / I_y if I_y > 0 else 0
q_mod_y = (M_service_x * (L_m / 2)) / I_x if I_x > 0 else 0

q_max = q_avg + q_mod_x + q_mod_y
q_min = q_avg - q_mod_x - q_mod_y

# 3. วิเคราะห์เสถียรภาพภายนอก (Geotechnical Stability Factors)
M_res_x = P_total_service * (L_m / 2)
M_ovr_x = M_service_x + (V_hy * Df_m)
FS_overturning_x = M_res_x / M_ovr_x if M_ovr_x > 0 else float('inf')

M_res_y = P_total_service * (B_m / 2)
M_ovr_y = M_service_y + (V_hx * Df_m)
FS_overturning_y = M_res_y / M_ovr_y if M_ovr_y > 0 else float('inf')

R_friction = P_total_service * base_friction
V_h_total = math.sqrt(V_hx**2 + V_hy**2)
FS_sliding = R_friction / V_h_total if V_h_total > 0 else float('inf')

# 4. แรงดันดินประลัยสำหรับออกแบบงานคอนกรีต (Ultimate Net Soil Pressure - Structural Level)
qu_base = (P_u * 1000) / (B_cm * L_cm) # หน่วย kg/cm²
qu_mod_x = (M_u_y * 1000 * 100 * (B_cm / 2)) / ((L_cm * (B_cm**3)) / 12)
qu_mod_y = (M_u_x * 1000 * 100 * (L_cm / 2)) / ((B_cm * (L_cm**3)) / 12)
qu_max_ksc = qu_base + qu_mod_x + qu_mod_y

# 5. ตรวจสอบกำลังรับแรงเฉือน (Structural Shear Safety - SDM)
phi_shear = 0.75
v_c_wide = 0.53 * math.sqrt(fc_prime)

# 5.1 แรงเฉือนคานกว้างทิศทาง X (Wide-Beam Shear X-Axis)
critical_x = ((B_cm - col_bx) / 2) - d_cm
V_u_wide_x = qu_max_ksc * L_cm * max(0.0, critical_x)
v_u_wide_x = V_u_wide_x / (L_cm * d_cm)

# 5.2 แรงเฉือนคานกว้างทิศทาง Y (Wide-Beam Shear Y-Axis)
critical_y = ((L_cm - col_by) / 2) - d_cm
V_u_wide_y = qu_max_ksc * B_cm * max(0.0, critical_y)
v_u_wide_y = V_u_wide_y / (B_cm * d_cm)

v_u_wide_max = max(v_u_wide_x, v_u_wide_y)

# 5.3 แรงเฉือนทะลุรอบเสาตอม่อ (Two-Way Punching Shear)
bo = 2 * ((col_bx + d_cm) + (col_by + d_cm))
area_punch = (col_bx + d_cm) * (col_by + d_cm)
V_u_punch = qu_max_ksc * ((B_cm * L_cm) - area_punch)
v_u_punch = V_u_punch / (bo * d_cm)

beta_c = max(col_bx, col_by) / min(col_bx, col_by)
v_c_p1 = 0.27 * (2 + 4/beta_c) * math.sqrt(fc_prime)
v_c_p2 = 0.27 * ((40 * d_cm / bo) + 2) * math.sqrt(fc_prime)
v_c_p3 = 1.06 * math.sqrt(fc_prime)
v_c_punch = min(v_c_p1, v_c_p2, v_c_p3)

# 6. ออกแบบปริมาณเหล็กเสริมต้านแรงดัดแยกทิศทาง (Flexural Reinforcement Design)
phi_flexure = 0.90
rho_min = 0.0018 if fy == 4000 else 0.0020

def calculate_rebar(M_u_critical, width_cm, effective_d):
    R_n = M_u_critical / (phi_flexure * width_cm * (effective_d ** 2))
    m_f = fy / (0.85 * fc_prime)
    if 1 - (2 * m_f * R_n / fy) > 0:
        rho_req = (1 / m_f) * (1 - math.sqrt(1 - (2 * m_f * R_n / fy)))
    else:
        rho_req = rho_min
    rho_final = max(rho_req, rho_min)
    return rho_final * width_cm * effective_d

# คำนวณโมเมนต์และเนื้อที่เหล็กเสริมแกน X และ Y ที่ขอบเสา
proj_x = (B_cm - col_bx) / 2
M_ux_critical = (qu_max_ksc * L_cm * (proj_x ** 2)) / 2
As_req_x = calculate_rebar(M_ux_critical, L_cm, d_cm)

proj_y = (L_cm - col_by) / 2
M_uy_critical = (qu_max_ksc * B_cm * (proj_y ** 2)) / 2
As_req_y = calculate_rebar(M_uy_critical, B_cm, d_cm)

# ตัวเลือกขนาดเหล็กมาตรฐานสำหรับงานอาคาร (DB16)
db_size = 16
as_single_bar = (math.pi / 4) * (db_size / 10) ** 2

num_bars_x = max(4, math.ceil(As_req_x / as_single_bar))
spacing_x = (L_cm - 15) / (num_bars_x - 1) if num_bars_x > 1 else 0

num_bars_y = max(4, math.ceil(As_req_y / as_single_bar))
spacing_y = (B_cm - 15) / (num_bars_y - 1) if num_bars_y > 1 else 0

# ตรวจสอบระยะฝังเหล็กหน่วงแรงดึง (Development Length)
L_d = (fy / (1.4 * math.sqrt(fc_prime))) * (db_size / 10)
available_L_d = min(proj_x, proj_y) - 7.5

# ==========================================
# PRESENTATION TABS SYSTEM
# ==========================================
tab_dash, tab_geo, tab_struct, tab_draw = st.tabs([
    "📊 แดชบอร์ดสรุปผลความปลอดภัย", 
    "🪨 รายการคำนวณด้านปฐพีกลศาสตร์", 
    "🧱 รายการคำนวณด้านงานโครงสร้างคอนกรีต", 
    "🎨 แบบขยายวิศวกรรมโครงสร้าง 2D"
])

with tab_dash:
    st.subheader("💡 ผลตรวจสอบดัชนีความปลอดภัยทางวิศวกรรม (Engineering Safety Indices)")
    mc1, mc2, mc3, mc4 = st.columns(4)
    
    with mc1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("แรงดันดินสูงสุดใต้ฐาน", f"{q_max:.2f} t/m²", f"ยอมให้ {qa_tsm:.1f}")
        status = "<span class='status-pass'>ผ่าน (PASS)</span>" if q_max <= qa_tsm else "<span class='status-fail'>ไม่ผ่าน (FAIL)</span>"
        st.markdown(f"สถานะดิน: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with mc2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("FS ต้านการพลิกคว่ำ (X)", f"{FS_overturning_x:.2f}", "เป้าหมาย ≥ 1.50")
        status = "<span class='status-pass'>ผ่าน (PASS)</span>" if FS_overturning_x >= 1.5 else "<span class='status-fail'>ไม่ผ่าน (FAIL)</span>"
        st.markdown(f"สถานะ: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with mc3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("FS ต้านการลื่นไถล", f"{FS_sliding:.2f}", "เป้าหมาย ≥ 1.50")
        status = "<span class='status-pass'>ผ่าน (PASS)</span>" if FS_sliding >= 1.5 else "<span class='status-fail'>ไม่ผ่าน (FAIL)</span>"
        st.markdown(f"สถานะ: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with mc4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("แรงเฉือนทะลุประสิทธิผล", f"{v_u_punch:.1f} ksc", f"ยอมให้ {phi_shear*v_c_punch:.1f}")
        status = "<span class='status-pass'>ผ่าน (PASS)</span>" if v_u_punch <= (phi_shear*v_c_punch) else "<span class='status-fail'>ไม่ผ่าน (FAIL)</span>"
        st.markdown(f"สถานะคอนกรีต: {status}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 ตารางสรุปข้อมูลสำหรับข้อกำหนดงานก่อสร้าง (Construction Specification)")
    st.info(f"📐 **ขนาดฐานรากแผ่:** กว้าง {B_m:.2f} ม. x ยาว {L_m:.2f} ม. | ความหนาทั้งหมด H = {H_cm:.0f} ซม. (ระยะ d = {d_cm:.1f} ซม.)")
    
    sc1, sc2 = st.columns(2)
    sc1.success(f"🧲 **เหล็กเสริมทิศทาง X:** เหล็กข้ออ้อย **DB16 จำนวน {num_bars_x} เส้น** ระยะห่างเฉลี่ย **@{spacing_x:.1f} ซม.**")
    sc2.success(f"🧲 **เหล็กเสริมทิศทาง Y:** เหล็กข้ออ้อย **DB16 จำนวน {num_bars_y} เส้น** ระยะห่างเฉลี่ย **@{spacing_y:.1f} ซม.**")

with tab_geo:
    st.markdown("### 🪨 บทวิเคราะห์การกระจายแรงดันและเสถียรภาพชั้นดิน")
    st.write("#### 1. การกระจายความเค้นแรงดันดินใช้งาน (Soil Bearing Pressure Distribution)")
    st.write(f"- น้ำหนักรวมแนวแกนที่กดลงดินชั้นล่าง (น้ำหนักโครงสร้าง + ฐานราก + ดินถมกลับ): `{P_total_service:.2f}` ตัน")
    st.write(f"- ระยะเยื้องศูนย์ที่เกิดขึ้นจริง: $e_x$ = `{e_x:.3f}` เมตร, $e_y$ = `{e_y:.3f}` เมตร")
    st.write(f"- ค่าขอบเขตพิกัดแกนรับแรงปลอดภัย (Kern Boundaries Area): $B/6$ = `{kern_x:.3f}` ม., $L/6$ = `{kern_y:.3f}` ม.")
    
    if is_within_kern:
        st.success(f"✔️ แรงลัพธ์ตกภายในพื้นที่ Kern ขอบเขตปลอดภัย ไม่เกิดสภาวะดินแยกตัวออกจากฐานราก ($q_{{min}}$ = `{q_min:.2f}` ตัน/ม²)")
    else:
        st.warning(f"⚠️ แรงลัพธ์อยู่นอกพิกัด Kern! เกิดแรงดึงใต้โครงสร้าง ($q_{{min}}$ = `{q_min:.2f}` ตัน/ม²) แนะนำให้เพิ่มขนาด B หรือ L เพื่อแก้ Soil Tension")
        
    st.write("#### 2. อัตราส่วนความปลอดภัยต้านทานแรงภายนอก (Geotechnical Stability Check)")
    st.write(f"- ค่าปัจจัยความปลอดภัยต่อการพลิกคว่ำทิศแกน X ($FS_{{overturning\ X}}$): `{FS_overturning_x:.2f}` (เกณฑ์เป้าหมาย $\ge 1.50$)")
    st.write(f"- ค่าปัจจัยความปลอดภัยต่อการพลิกคว่ำทิศแกน Y ($FS_{{overturning\ Y}}$): `{FS_overturning_y:.2f}` (เกณฑ์เป้าหมาย $\ge 1.50$)")
    st.write(f"- ค่าปัจจัยความปลอดภัยต้านการลื่นไถลแนวราบใต้ท้องฐาน ($FS_{{sliding}}$): `{FS_sliding:.2f}` (เกณฑ์เป้าหมาย $\ge 1.50$)")

with tab_struct:
    st.markdown("### 🧱 รายการตรวจวัดกำลังวัสดุคอนกรีตเสริมเหล็กด้วยวิธีประลัย (SDM)")
    
    st.write("#### 1. การตรวจสอบแรงเฉือนแบบคานกว้างคู่ (Two-Directional One-Way Shear Check)")
    st.write(f"- หน่วยแรงเฉือนคานกว้างวิกฤตที่เกิดจริงในทิศแกน X ($v_{{ux}}$) = `{v_u_wide_x:.2f}` kg/cm²")
    st.write(f"- หน่วยแรงเฉือนคานกว้างวิกฤตที่เกิดจริงในทิศแกน Y ($v_{{uy}}$) = `{v_u_wide_y:.2f}` kg/cm²")
    st.write(f"- พิกัดกำลังต้านทานแรงเฉือนคานกว้างสูงสุดที่คอนกรีตยอมให้ ($\phi v_c$) = `{phi_shear * v_c_wide:.2f}` kg/cm²")
    if v_u_wide_max <= (phi_shear * v_c_wide):
        st.success("✔️ ผ่าน: ความหนาฐานคอนกรีตเพียงพอต่อการต้านแรงเฉือนคานกว้างโดยไม่ต้องใส่เหล็กแกนพิเศษ")
    else:
        st.error("❌ วิกฤต: คอนกรีตบางเกินไปจะพังทลายเนื่องจากแรงเฉือนคานกว้าง กรุณาเพิ่มความหนา H ด่วน!")

    st.write("#### 2. การตรวจสอบแรงเฉือนทะลุรอบเสาตอม่อ (Two-Way Punching Shear Validation)")
    st.write(f"- ความยาวแนวเส้นรอบรูปวิกฤตห่างผิวเสาออกมา $d/2$ ($b_0$) = `{bo:.1f}` ซม.")
    st.write(f"- หน่วยแรงเฉือนทะลุประลัยที่เกิดขึ้นจริงรอบเสา ($v_u$) = `{v_u_punch:.2f}` kg/cm²")
    st.write(f"- พิกัดกำลังต้านแรงเฉือนทะลุที่ยอมให้ต่ำสุดจาก 3 สมการควบคุม ($\phi v_c$) = `{phi_shear * v_c_punch:.2f}` kg/cm²")
    if v_u_punch <= (phi_shear * v_c_punch):
        st.success("✔️ ผ่าน: หน้าตัดคอนกรีตสามารถกระจายแรงและต้านแรงเฉือนทะลุได้อย่างสมบูรณ์")
    else:
        st.error("❌ วิกฤต: เกิดแรงเฉือนทะลุวิบัติ เสาตอม่อจะเจาะทะลุแผ่นฐานราก ต้องเพิ่มความหนา H")

    st.write("#### 3. การตรวจสอบระยะฝังเหล็กเสริม (Development Length Check)")
    st.write(f"- ระยะยึดเหนี่ยวรั้งเหล็กเสริมแกนที่จำเป็นจริงสำหรับเหล็ก DB16 ($L_d$) = `{L_d:.1f}` ซม.")
    st.write(f"- ระยะของปีกฐานรากที่พร้อมให้เหล็กฝังตัวแกนเหนี่ยวรั้งจริง ($L_{{available}}$) = `{available_L_d:.1f}` ซม.")
    if available_L_d >= L_d:
        st.success("✔️ ผ่าน: ระยะปีกฐานรากยาวพอที่จะยึดเหนี่ยวเหล็กหลัก ไม่ต้องทำการพับงอขอพิเศษสำหรับจุดนี้")
    else:
        st.warning(f"⚠️ เตือน: ระยะฝังไม่พอ ({available_L_d:.1f} ซม. < {L_d:.1f} ซม.) ช่างต้องงอขอ 90 องศามาตรฐานที่ปลายเหล็ก")

with tab_draw:
    st.markdown("### 🎨 แบบขยายงานวิศวกรรมฐานรากแผ่ (Structural Simulation Blueprint)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    
    # DRAWING 1: PLAN VIEW
    ax1.add_patch(plt.Rectangle((0, 0), B_m, L_m, color='#F1F5F9', ec='#1E3A8A', lw=2.5, label='Footing Area'))
    cx = (B_m - (col_bx/100)) / 2
    cy = (L_m - (col_by/100)) / 2
    ax1.add_patch(plt.Rectangle((cx, cy), col_bx/100, col_by/100, color='#FEE2E2', ec='#DC2626', lw=2, label='Column'))
    
    p_off = (d_cm / 100) / 2
    ax1.add_patch(plt.Rectangle((cx - p_off, cy - p_off), (col_bx/100) + 2*p_off, (col_by/100) + 2*p_off, fill=False, ec='#D97706', lw=1.5, ls='--', label='Punching Perimeter (d/2)'))
    
    # วาดเหล็กปลอกตะแกรงทิศทาง X และ Y
    for i in range(min(num_bars_x, 10)):
        pos = 0.075 + i * ((L_m - 0.15) / (min(num_bars_x, 10) - 1))
        ax1.plot([0.075, B_m - 0.075], [pos, pos], color='#3B82F6', lw=1.2, alpha=0.8)
    for i in range(min(num_bars_y, 10)):
        pos = 0.075 + i * ((B_m - 0.15) / (min(num_bars_y, 10) - 1))
        ax1.plot([pos, pos], [0.075, L_m - 0.075], color='#1D4ED8', lw=1.2, alpha=0.8)

    ax1.set_xlim(-0.3, B_m + 0.3)
    ax1.set_ylim(-0.3, L_m + 0.3)
    ax1.set_aspect('equal')
    ax1.set_title("แปลนขยายเหล็กเสริมตะแกรงล่าง (PLAN VIEW)", fontsize=11, fontweight='bold', color='#1E3A8A')
    ax1.axis('off')
    ax1.text(B_m/2, L_m + 0.05, f"L = {L_m:.2f} m", ha='center', va='bottom', weight='bold')
    ax1.text(B_m + 0.05, L_m/2, f"B = {B_m:.2f} m", ha='left', va='center', weight='bold', rotation=-90)
    ax1.legend(loc='lower left', fontsize=8)

    # DRAWING 2: SECTION VIEW
    ax2.plot([-0.4, B_m + 0.4], [Df_m, Df_m], color='#78350F', lw=2, ls='-', label='Natural Ground Level')
    ax2.add_patch(plt.Rectangle((-0.04, 0.04), B_m + 0.08, 0.04, color='#CBD5E1', ec='#64748B', lw=1)) # คอนกรีตหยาบ
    ax2.add_patch(plt.Rectangle((0, 0.08), B_m, H_cm/100, color='#E2E8F0', ec='#1E3A8A', lw=2.5)) # ตัวฐานราก
    ax2.add_patch(plt.Rectangle((cx, 0.08 + H_cm/100), col_bx/100, Df_m - (0.08 + H_cm/100) + 0.3, color='#FEE2E2', ec='#DC2626', lw=2)) # เสา
    
    # เส้นเหล็กเสริมตะแกรงพร้อมพับงอขอตั้งขึ้น
    ry = 0.08 + 0.075
    ax2.plot([0.075, B_m - 0.075], [ry, ry], color='#3B82F6', lw=2.5)
    ax2.plot([0.075, 0.075], [ry, ry + 0.12], color='#3B82F6', lw=2.5)
    ax2.plot([B_m - 0.075, B_m - 0.075], [ry, ry + 0.12], color='#3B82F6', lw=2.5)
    
    # เหล็กเดือยรากเสา (Dowels)
    ax2.plot([cx + 0.04, cx + 0.04], [0.08 + 0.075, Df_m + 0.15], color='#DC2626', lw=2)
    ax2.plot([cx + (col_bx/100) - 0.04, cx + (col_bx/100) - 0.04], [0.08 + 0.075, Df_m + 0.15], color='#DC2626', lw=2)

    ax2.set_xlim(-0.4, B_m + 0.4)
    ax2.set_ylim(-0.1, Df_m + 0.5)
    ax2.set_title("รูปตัดแสดงโครงสร้างชั้นดินและระยะฝัง (SECTION VIEW)", fontsize=11, fontweight='bold', color='#1E3A8A')
    ax2.axis('off')
    ax2.text(B_m + 0.03, 0.08 + (H_cm/200), f"H = {H_cm:.0f} cm", va='center', weight='bold')
    ax2.text(-0.05, Df_m, f"Df = {Df_m:.2f} m", ha='right', va='center', color='#78350F', weight='bold')

    st.pyplot(fig)
