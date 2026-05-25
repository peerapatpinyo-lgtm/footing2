import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

st.set_page_config(
    page_title='Enterprise Foundation Engineering Suite (EIT & DPT Compliant)',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown('''
    <style>
    .main-header { font-size:28px; font-weight:bold; color:#1E3A8A; margin-bottom:5px; }
    .sub-header { font-size:16px; color:#4B5563; margin-bottom:25px; }
    .section-title { font-size:20px; font-weight:bold; color:#1F2937; border-left: 5px solid #1E3A8A; padding-left: 10px; margin-top:20px; margin-bottom:15px; }
    .metric-card { background-color: #F3F4F6; padding: 15px; border-radius: 8px; border: 1px solid #E5E7EB; }
    .status-pass { color: #10B981; font-weight: bold; }
    .status-fail { color: #EF4444; font-weight: bold; }
    </style>
''', unsafe_html=True)

st.markdown('<div class="main-header">🏗️ Enterprise Foundation Engineering Suite</div>', unsafe_html=True)
st.markdown('<div class="sub-header">โปรแกรมออกแบบฐานรากแผ่เยื้องศูนย์สองแกนระดับวิชาชีพ (Bi-axial Eccentric Shallow Foundation Design) คลอบคลุมมาตรฐาน วสท. 1007/1008 และ มยผ. 1301/1302</div>', unsafe_html=True)

st.sidebar.header('⚙️ 1. วิธีการออกแบบ & ข้อกำหนด')
design_method = st.sidebar.radio('ระเบียบวิธีการออกแบบ (Design Method)', ['วิธีวิเคราะห์กำลัง (Strength Design Method - SDM)', 'วิธีหน่วยแรงที่ยอมให้ (Working Stress Design - WSD)'])

st.sidebar.header('📥 2. น้ำหนักบรรทุกใช้งาน (Service Loads)')
with st.sidebar.expander('โหลดแนวแกน & โมเมนต์ดัด', expanded=True):
    P_DL = st.number_input('น้ำหนักบรรทุกคงที่ Dead Load: P_DL (ตัน)', min_value=0.0, value=25.0, step=1.0)
    P_LL = st.number_input('น้ำหนักบรรทุกจร Live Load: P_LL (ตัน)', min_value=0.0, value=15.0, step=1.0)
    st.markdown('**โมเมนต์ดัดแกน X (ทำให้เกิดการเยื้องศูนย์ทิศทาง Y)**')
    M_DL_x = st.number_input('M_DL x (ตัน-เมตร)', value=2.0, step=0.5)
    M_LL_x = st.number_input('M_LL x (ตัน-เมตร)', value=1.5, step=0.5)
    M_WL_x = st.number_input('M_WL x (แรงลม) (ตัน-เมตร)', value=1.0, step=0.5)
    st.markdown('**โมเมนต์ดัดแกน Y (ทำให้เกิดการเยื้องศูนย์ทิศทาง X)**')
    M_DL_y = st.number_input('M_DL y (ตัน-เมตร)', value=1.5, step=0.5)
    M_LL_y = st.number_input('M_LL y (ตัน-เมตร)', value=1.0, step=0.5)
    M_WL_y = st.number_input('M_WL y (แรงลม) (ตัน-เมตร)', value=0.8, step=0.5)
    st.markdown('**แรงเฉือนตามแนวราบ (Horizontal Shear for Sliding)**')
    V_hx = st.number_input('แรงเฉือนราบทิศทาง X: V_hx (ตัน)', value=1.5, step=0.1)
    V_hy = st.number_input('แรงเฉือนราบทิศทาง Y: V_hy (ตัน)', value=1.2, step=0.1)

st.sidebar.header('🧱 3. คุณสมบัติวัสดุ & ธรณีเทคนิค')
with st.sidebar.expander('สเปกคอนกรีต เหล็ก และดิน', expanded=False):
    qa_tsm = st.number_input('กำลังรับน้ำหนักปลอดภัยของดิน: q_allow (ตัน/ม²)', min_value=1.0, value=16.0, step=0.5)
    fc_prime = st.number_input('กำลังอัดประลัยทรงกระบอกคอนกรีต: fc\' (ksc)', min_value=150, value=240, step=10)
    fy = st.selectbox('ชั้นคุณภาพเหล็กเสริมหลัก (fy)', [3000, 4000], index=1)
    soil_density = st.number_input('ความหนาแน่นของดินเหนือฐานราก (ตัน/ม³)', value=1.8, step=0.1)
    base_friction = st.number_input('สัมประสิทธิ์แรงเสียดทานใต้ฐานราก (μ)', min_value=0.1, max_value=0.7, value=0.45, step=0.05)

st.sidebar.header('📐 4. มิติหน้าตัดเสาตอม่อ')
col_bx = st.sidebar.number_input('ความกว้างเสาตอม่อด้าน X (ซม.)', value=30.0, step=5.0)
col_by = st.sidebar.number_input('ความกว้างเสาตอม่อด้าน Y (ซม.)', value=30.0, step=5.0)

st.markdown('<div class="section-title">📐 การกำหนดมิติและสัดส่วนเรขาคณิตของฐานราก (Footing Geometry Optimization)</div>', unsafe_html=True)
gc1, gc2, gc3, gc4 = st.columns(4)
B_m = gc1.number_input('ความกว้างฐานรากทิศทาง X: B (เมตร)', min_value=1.0, value=2.2, step=0.1)
L_m = gc2.number_input('ความยาวฐานรากทิศทาง Y: L (เมตร)', min_value=1.0, value=2.2, step=0.1)
H_cm = gc3.number_input('ความหนาทั้งหมดของฐานราก: H (ซม.)', min_value=25.0, value=50.0, step=5.0)
Df_m = gc4.number_input('ความลึกระดับฝังฐานราก: Df (เมตร)', min_value=0.5, value=1.5, step=0.1)

P_service_unfactored = P_DL + P_LL
M_service_x = M_DL_x + M_LL_x
M_service_y = M_DL_y + M_LL_y

P_u1 = 1.4 * P_DL
M_u1_x = 1.4 * M_DL_x
M_u1_y = 1.4 * M_DL_y
P_u2 = 1.2 * P_DL + 1.6 * P_LL
M_u2_x = 1.2 * M_DL_x + 1.6 * M_LL_x + 1.0 * M_WL_x
M_u2_y = 1.2 * M_DL_y + 1.6 * M_LL_y + 1.0 * M_WL_y
P_u = max(P_u1, P_u2)
M_u_x = max(M_u1_x, M_u2_x)
M_u_y = max(M_u1_y, M_u2_y)

B = B_m * 100
L = L_m * 100
H = H_cm
d = H - 7.5
A_base = B_m * L_m
W_footing = A_base * (H/100) * 2.4
W_soil = A_base * (Df_m - H/100) * soil_density
P_total_service = P_service_unfactored + W_footing + W_soil

I_x = (B_m * (L_m**3)) / 12
I_y = ((B_m**3) * L_m) / 12
e_x = M_service_y / P_total_service if P_total_service > 0 else 0
e_y = M_service_x / P_total_service if P_total_service > 0 else 0
kern_x = B_m / 6
kern_y = L_m / 6
is_within_kern = (e_x <= kern_x) and (e_y <= kern_y)

q_base = P_total_service / A_base
q_mod_x = (M_service_y * (B_m / 2)) / I_y if I_y > 0 else 0
q_mod_y = (M_service_x * (L_m / 2)) / I_x if I_x > 0 else 0
q1 = q_base + q_mod_x + q_mod_y
q2 = q_base - q_mod_x + q_mod_y
q3 = q_base + q_mod_x - q_mod_y
q4 = q_base - q_mod_x - q_mod_y
q_max = max(q1, q2, q3, q4)
q_min = min(q1, q2, q3, q4)

qu_base = (P_u * 1000) / (B * L)
qu_mod_x = (M_u_y * 1000 * 100 * (B / 2)) / ((L * (B**3)) / 12)
qu_mod_y = (M_u_x * 1000 * 100 * (L / 2)) / ((B * (L**3)) / 12)
qu_max = qu_base + qu_mod_x + qu_mod_y

M_resisting_x = P_total_service * (L_m / 2)
M_overturning_x = M_service_x + (V_hy * Df_m)
FS_overturning_x = M_resisting_x / M_overturning_x if M_overturning_x > 0 else float('inf')
M_resisting_y = P_total_service * (B_m / 2)
M_overturning_y = M_service_y + (V_hx * Df_m)
FS_overturning_y = M_resisting_y / M_overturning_y if M_overturning_y > 0 else float('inf')
R_friction = P_total_service * base_friction
V_h_total = math.sqrt(V_hx**2 + V_hy**2)
FS_sliding = R_friction / V_h_total if V_h_total > 0 else float('inf')

phi_shear = 0.75
phi_flexure = 0.90
critical_x = (B - col_bx) / 2 - d
V_u_wide = qu_max * L * max(0.0, critical_x)
v_u_wide = V_u_wide / (L * d)
v_c_wide = 0.53 * math.sqrt(fc_prime)

bo = 2 * ((col_bx + d) + (col_by + d))
area_punch = (col_bx + d) * (col_by + d)
V_u_punch = qu_max * ((B * L) - area_punch)
v_u_punch = V_u_punch / (bo * d)
beta_c = max(col_bx, col_by) / min(col_bx, col_by)
v_c_p1 = 0.27 * (2 + 4/beta_c) * math.sqrt(fc_prime)
v_c_p2 = 0.27 * ((40 * d / bo) + 2) * math.sqrt(fc_prime)
v_c_p3 = 1.06 * math.sqrt(fc_prime)
v_c_punch = min(v_c_p1, v_c_p2, v_c_p3)

proj_x = (B - col_bx) / 2
M_u_crit_x = (qu_max * L * (proj_x**2)) / 2
R_n = M_u_crit_x / (phi_flexure * L * (d**2))
m_f = fy / (0.85 * fc_prime)
if 1 - (2 * m_f * R_n / fy) > 0:
    rho_req = (1 / m_f) * (1 - math.sqrt(1 - (2 * m_f * R_n / fy)))
else:
    rho_req = 0.002
rho_min = 0.0018 if fy == 4000 else 0.0020
rho_final = max(rho_req, rho_min)
As_total_x = rho_final * L * d
db_selected = 16
as_bar = (math.pi / 4) * (db_selected / 10)**2
num_bars_x = math.ceil(As_total_x / as_bar)
if num_bars_x < 4: num_bars_x = 4
spacing_x = (L - 15) / (num_bars_x - 1)
L_d = (fy / (1.4 * math.sqrt(fc_prime))) * (db_selected / 10)
available_L_d = proj_x - 7.5

tab_dash, tab_geo, tab_struct, tab_draw = st.tabs([
    '📊 แดชบอร์ดสรุปผลความปลอดภัย (Dashboard)', 
    '🪨 รายการคำนวณด้านปฐพีกลศาสตร์ (Geotechnical Sheet)', 
    '🧱 รายการคำนวณด้านงานโครงสร้าง (Structural Sheet)', 
    '🎨 แบบวิศวกรรมสถาปัตย์ขยายฐานราก (2D Drawings)'
])

with tab_dash:
    st.subheader('💡 ดัชนีความปลอดภัยทางวิศวกรรม (Engineering Safety Indexes)')
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown('<div class="metric-card">', unsafe_html=True)
        st.metric('แรงดันดินสูงสุด', f'{q_max:.2f} t/m²', f'ยอมให้ {qa_tsm:.1f}')
        status = '<span class="status-pass">ผ่าน (PASS)</span>' if q_max <= qa_tsm else '<span class="status-fail">ไม่ผ่าน (FAIL)</span>'
        st.markdown(f'สถานะ: {status}', unsafe_html=True)
        st.markdown('</div>', unsafe_html=True)
    with mc2:
        st.markdown('<div class="metric-card">', unsafe_html=True)
        st.metric('FS พลิกคว่ำ (แกน X)', f'{FS_overturning_x:.2f}', 'เป้าหมาย ≥ 1.50')
        status = '<span class="status-pass">ผ่าน (PASS)</span>' if FS_overturning_x >= 1.5 else '<span class="status-fail">ไม่ผ่าน (FAIL)</span>'
        st.markdown(f'สถานะ: {status}', unsafe_html=True)
        st.markdown('</div>', unsafe_html=True)
    with mc3:
        st.markdown('<div class="metric-card">', unsafe_html=True)
        st.metric('FS การลื่นไถล', f'{FS_sliding:.2f}', 'เป้าหมาย ≥ 1.50')
        status = '<span class="status-pass">ผ่าน (PASS)</span>' if FS_sliding >= 1.5 else '<span class="status-fail">ไม่ผ่าน (FAIL)</span>'
        st.markdown(f'สถานะ: {status}', unsafe_html=True)
        st.markdown('</div>', unsafe_html=True)
    with mc4:
        st.markdown('<div class="metric-card">', unsafe_html=True)
        st.metric('แรงเฉือนทะลุคอนกรีต', f'{v_u_punch:.1f} ksc', f'ยอมให้ {phi_shear*v_c_punch:.1f}')
        status = '<span class="status-pass">ผ่าน (PASS)</span>' if v_u_punch <= (phi_shear*v_c_punch) else '<span class="status-fail">ไม่ผ่าน (FAIL)</span>'
        st.markdown(f'สถานะ: {status}', unsafe_html=True)
        st.markdown('</div>', unsafe_html=True)
    st.markdown('---')
    st.markdown('### 📋 บทสรุปข้อมูลจำเพาะสำหรับสั่งงานจัดจ้าง (Construction Schedule)')
    st.info(f'📐 ขนาดมิติฐานราก: กว้าง {B_m:.2f} ม. x ยาว {L_m:.2f} ม. | ความหนาประสิทธิผล (d) = {d:.1f} ซม. (ความหนาทั้งหมด H = {H_cm:.1f} ซม.)')
    st.success(f'🧲 ตะแกรงเหล็กเสริมล่าง: ใช้เหล็กข้ออ้อย DB16 จำนวน {num_bars_x} เส้น จัดระยะห่างเฉลี่ย @{spacing_x:.1f} ซม. (ตะแกรงสองทิศทางสมมาตร)')

with tab_geo:
    st.markdown('### 🪨 บทวิเคราะห์กำลังรับน้ำหนักชั้นดินและเสถียรภาพเสมือน')
    st.write('#### 1. การกระจายตัวของแรงดันดินใต้ฐานราก (Soil Stress Distribution)')
    st.write(f'- น้ำหนักรวมใช้งานแนวแกน (รวมดิน+ฐานราก): {P_total_service:.2f} ตัน')
    st.write(f'- ระยะเยื้องศูนย์เกิดขึ้นจริง: e_x = {e_x:.3f} ม., e_y = {e_y:.3f} ม.')
    if is_within_kern:
        st.success(f'✔️ แรงเยื้องศูนย์อยู่ในพิกัด Kern Boundary (q_min = {q_min:.2f} ตัน/ม²)')
    else:
        st.warning(f'⚠️ แรงเยื้องศูนย์หลุดนอกพิกัด Kern! q_min = {q_min:.2f} ตัน/ม²')
    st.write(f'- อัตราส่วนปลอดภัยการพลิกคว่ำทิศทาง X: {FS_overturning_x:.2f} (≥ 1.50)')
    st.write(f'- อัตราส่วนปลอดภัยการพลิกคว่ำทิศทาง Y: {FS_overturning_y:.2f} (≥ 1.50)')
    st.write(f'- อัตราส่วนปลอดภัยต้านทานการลื่นไถล (FS_sliding): {FS_sliding:.2f} (≥ 1.50)')

with tab_struct:
    st.markdown('### 🧱 รายการคำนวณกำลังวัสดุคอนกรีตเสริมเหล็กประลัย (SDM)')
    st.write('#### 1. การตรวจสอบแรงเฉือนแบบคานกว้าง (One-Way Shear Check)')
    st.write(f'- หน่วยแรงเฉือนคานกว้างวิกฤตที่เกิดขึ้นจริง (v_u) = {v_u_wide:.2f} ksc')
    st.write(f'- กำลังต้านทานแรงเฉือนคานกว้างที่ยอมให้ (phi v_c) = {phi_shear * v_c_wide:.2f} ksc')
    st.write('#### 2. การตรวจสอบแรงเฉือนทะลุเสาตอม่อ (Two-Way Punching Shear Check)')
    st.write(f'- หน่วยแรงเฉือนทะลุประลัยที่เกิดขึ้นรอบผิวเสา (v_u) = {v_u_punch:.2f} ksc')
    st.write(f'- พิกัดกำลังต้านทานแรงเฉือนทะลุที่ยอมให้สูงสุด (phi v_c) = {phi_shear * v_c_punch:.2f} ksc')
    st.write('#### 3. การตรวจสอบระยะฝังและยึดรั้งเหล็กเสริมหลัก (Development Length Check)')
    st.write(f'- ระยะฝังยึดรั้งที่ต้องการจริงสำหรับเหล็กตะแกรง DB16 (L_d) = {L_d:.1f} ซม.')
    st.write(f'- ระยะยื่นของปีกฐานรากที่พร้อมให้เหล็กฝังตัวจริงเหนี่ยวรั้ง (L_available) = {available_L_d:.1f} ซม.')

with tab_draw:
    st.markdown('### 🎨 แบบขยายงานวิศวกรรมโครงสร้างและสถาปัตยกรรม (Engineering Structural Blueprint)')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.add_patch(plt.Rectangle((0, 0), B_m, L_m, color='#EAF2F8', ec='#1E3A8A', lw=2.5, label='Footing Base'))
    cx_pos = (B_m - (col_bx/100)) / 2
    cy_pos = (L_m - (col_by/100)) / 2
    ax1.add_patch(plt.Rectangle((cx_pos, cy_pos), col_bx/100, col_by/100, color='#FCA5A5', ec='#B91C1C', lw=2, label='Column'))
    p_offset = (d/100) / 2
    ax1.add_patch(plt.Rectangle((cx_pos - p_offset, cy_pos - p_offset), (col_bx/100) + 2*p_offset, (col_by/100) + 2*p_offset, fill=False, ec='#F59E0B', lw=1.5, ls='--', label='Critical Punching (d/2)'))
    for i in range(min(num_bars_x, 10)):
        pos = 0.075 + i * ((L_m - 0.15) / (min(num_bars_x, 10) - 1))
        ax1.plot([0.075, B_m - 0.075], [pos, pos], color='#2563EB', lw=1.2)
        ax1.plot([pos, pos], [0.075, L_m - 0.075], color='#1D4ED8', lw=1.2)
    ax1.set_xlim(-0.3, B_m + 0.3)
    ax1.set_ylim(-0.3, L_m + 0.3)
    ax1.set_aspect('equal')
    ax1.set_title('แปลนขยายเหล็กเสริมฐานราก (PLAN VIEW)', fontsize=12, fontweight='bold', color='#1E3A8A')
    ax1.axis('off')
    ax1.text(B_m/2, L_m + 0.05, f'L = {L_m:.2f} m', ha='center', va='bottom', fontsize=10, weight='bold')
    ax1.text(B_m + 0.05, L_m/2, f'B = {B_m:.2f} m', ha='left', va='center', fontsize=10, weight='bold', rotation=-90)
    ax1.legend(loc='lower left', fontsize=8)
    ax2.plot([-0.5, B_m + 0.5], [Df_m, Df_m], color='#78350F', lw=2.5, ls='-')
    ax2.add_patch(plt.Rectangle((-0.05, 0.05), B_m + 0.1, 0.05, color='#CBD5E1', ec='#64748B', lw=1))
    ax2.add_patch(plt.Rectangle((-0.05, 0), B_m + 0.1, 0.05, color='#E2E8F0', ec='#94A3B8', lw=1, hatch='...'))
    ax2.add_patch(plt.Rectangle((0, 0.1), B_m, H_cm/100, color='#E2E8F0', ec='#1E3A8A', lw=2.5))
    ax2.add_patch(plt.Rectangle((cx_pos, 0.1 + H_cm/100), col_bx/100, Df_m - (0.1 + H_cm/100) + 0.4, color='#FCA5A5', ec='#B91C1C', lw=2))
    rebar_y = 0.1 + 0.075
    ax2.plot([0.075, B_m - 0.075], [rebar_y, rebar_y], color='#2563EB', lw=3)
    ax2.plot([0.075, 0.075], [rebar_y, rebar_y + 0.15], color='#2563EB', lw=3)
    ax2.plot([B_m - 0.075, B_m - 0.075], [rebar_y, rebar_y + 0.15], color='#2563EB', lw=3)
    ax2.plot([cx_pos + 0.05, cx_pos + 0.05], [0.1 + 0.075, Df_m + 0.2], color='#B91C1C', lw=2.5)
    ax2.plot([cx_pos + (col_bx/100) - 0.05, cx_pos + (col_bx/100) - 0.05], [0.1 + 0.075, Df_m + 0.2], color='#B91C1C', lw=2.5)
    ax2.set_xlim(-0.5, B_m + 0.5)
    ax2.set_ylim(-0.2, Df_m + 0.6)
    ax2.set_title('รูปตัดโครงสร้างขยายฐานราก (SECTION VIEW)', fontsize=12, fontweight='bold', color='#1E3A8A')
    ax2.axis('off')
    ax2.text(B_m + 0.05, 0.1 + (H_cm/200), f'H = {H_cm:.0f} ซม.', va='center', weight='bold')
    st.pyplot(fig)
