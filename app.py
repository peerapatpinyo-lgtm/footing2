import streamlit as st
import math
import matplotlib.pyplot as plt

# ตั้งค่าหน้าเว็บแบบกว้างพิเศษ
st.set_page_config(page_title="Ultimate Footing Expert (SDM)", layout="wide")

st.title("🏗️ Ultimate Foundation Designer (วิธี SDM)")
st.caption("ระบบคำนวณฐานรากแผ่รองรับแรงเยื้องศูนย์สองทิศทาง (Bi-axial Shear & Moment) ตามมาตรฐาน มยผ. / ACI 318")
st.markdown("---")

# ==========================================
# SIDEBAR: INPUT PARAMETERS (แยก DL / LL ชัดเจน)
# ==========================================
st.sidebar.header("📥 1. น้ำหนักบรรทุก (Loads)")
with st.sidebar.expander("📊 Axial & Moments", expanded=True):
    P_dl = st.number_input("Dead Load: P_DL (ตัน)", min_value=0.0, value=20.0)
    P_ll = st.number_input("Live Load: P_LL (ตัน)", min_value=0.0, value=15.0)
    Mx_ser = st.number_input("Moment X: M_x Service (ตัน-เมตร)", value=2.5)
    My_ser = st.number_input("Moment Y: M_y Service (ตัน-เมตร)", value=1.5)

st.sidebar.header("🧱 2. คุณสมบัติวัสดุ & ดิน")
with st.sidebar.expander("สเปกวัสดุตามข้อกำหนดไทย", expanded=False):
    qa_tsm = st.number_input("กำลังรับน้ำหนักดินปลอดภัย: qa (ตัน/ม²)", min_value=1.0, value=18.0)
    fc_prime = st.number_input("กำลังอัดคอนกรีตประลัย: fc' (ksc)", min_value=100, value=280)
    fy = st.selectbox("ชั้นคุณภาพเหล็กเสริม: fy (ksc)", options=[3000, 4000], index=1) # SD30 หรือ SD40

st.sidebar.header("📐 3. ขนาดหน้าตัดเสาตอม่อ")
col_b = st.sidebar.number_input("ความกว้างเสา b (ซม.)", value=30.0)
col_t = st.sidebar.number_input("ความลึกเสา t (ซม.)", value=30.0)

# ==========================================
# ENGINEERING CALCULATION ENGINE (SDM)
# ==========================================

# 1. คำนวณน้ำหนักบรรทุก
P_service = P_dl + P_ll
# ตัวคูณกำลังตามมาตรฐาน มยผ. ล่าสุด (1.2DL + 1.6LL)
P_ultimate = (1.2 * P_dl) + (1.6 * P_ll)
M_u_x = 1.6 * Mx_ser  # สมมติรวมในส่วนแปรผัน
M_u_y = 1.6 * My_ser

# ตัวลดกำลัง (Reduction Factors) ตามวิธี SDM
phi_bending = 0.90
phi_shear = 0.75

# 2. ประมาณการขนาดฐานรากเบื้องต้น (พิจารณาแรงเยื้องศูนย์เบื้องต้น เผื่อพื้นที่เพิ่ม 30%)
area_req = (P_service * 1.30) / qa_tsm
B_init = math.ceil(math.sqrt(area_req) * 10) / 10
if B_init < 1.0: B_init = 1.0

# ส่วนเลือกขนาดจริงบนหน้าเว็บเพื่อความยืดหยุ่นของวิศวกร
st.markdown("### 🛠️ ปรับเปลี่ยนขนาดมิติฐานรากเพื่อทดสอบระบบ (Interactive Optimization)")
c1, c2, c3 = st.columns(3)
B_input = c1.number_input("ระบุความกว้างฐานราก B (เมตร)", min_value=1.0, value=float(B_init), step=0.1)
L_input = c2.number_input("ระบุความยาวฐานราก L (เมตร)", min_value=1.0, value=float(B_init), step=0.1)
H_input = c3.number_input("ระบุความหนาทั้งหมด H (ซม.)", min_value=20.0, value=45.0, step=5.0)

# แปลงหน่วยเข้าสู่ระบบคำนวณ (kg, cm)
B = B_input * 100
L = L_input * 100
H = H_input
d = H - 7.5 # สมมติระยะหุ้มเหล็ก 7.5 ซม.

# 3. ตรวจสอบแรงดันดิน (Geotechnical Check - ใช้ Service Load)
# สูตรแรงดันดินเยื้องศูนย์สองทิศทาง: q = P/A +- 6Mx/(B^2*L) +- 6My/(B*L^2)
A = B * L
S_x = (B * B * L) / 6
S_y = (B * L * L) / 6

P_kg = P_service * 1000
Mx_kgcm = Mx_ser * 1000 * 100
My_kgcm = My_ser * 1000 * 100

q_avg = P_kg / A
q_stress_x = Mx_kgcm / S_x if S_x > 0 else 0
q_stress_y = My_kgcm / S_y if S_y > 0 else 0

q_max = q_avg + q_stress_x + q_stress_y
q_min = q_avg - q_stress_x - q_stress_y

# แปลงกลับเป็น ตัน/มตร.ม. เพื่อแสดงผล
q_max_tsm = q_max * 10
q_min_tsm = q_min * 10

# 4. คำนวณแรงดันดินประลัย (Ultimate Net Pressure) สำหรับออกแบบโครงสร้างคอนกรีต
P_u_kg = P_ultimate * 1000
Mu_x_kgcm = M_u_x * 1000 * 100
Mu_y_kgcm = M_u_y * 1000 * 100

qu_max = (P_u_kg / A) + (Mu_x_kgcm / S_x) + (Mu_y_kgcm / S_y)
qu_min = (P_u_kg / A) - (Mu_x_kgcm / S_x) - (Mu_y_kgcm / S_y)
qu_design = max(qu_max, qu_min) # ใช้แรงดันสูงสุดเพื่อความปลอดภัยในการออกแบบเหล็กและแรงเฉือน

# 5. ตรวจสอบแรงเฉือนตามวิธี SDM
# กำลังรับแรงเฉือนของคอนกรีต (ตามสูตร ACI/มยผ. หน่วย ksc)
v_c_wide = 0.53 * math.sqrt(fc_prime)
v_c_punch = 1.06 * math.sqrt(fc_prime)

# แรงเฉือนคานกว้างวิกฤต (คิดที่ระยะ d จากขอบเสา ทิศทางด้านสั้น)
x_critical = (B - col_b) / 2
V_u_wide = qu_design * L * (x_critical - d)
v_u_wide = V_u_wide / (L * d) / phi_shear

# แรงเฉือนทะลุวิกฤต (คิดที่เส้นรอบรูป d/2 รอบเสา)
b0 = 2 * ((col_b + d) + (col_t + d))
area_punch = (col_b + d) * (col_t + d)
V_u_punch = qu_design * ((B * L) - area_punch)
v_u_punch = V_u_punch / (b0 * d) / phi_shear

# 6. ออกแบบเหล็กเสริมต้านแรงดัด (SDM Flexural Design)
M_u_critical = (qu_design * L * (x_critical ** 2)) / 2 # kg-cm
# หาปริมาณเหล็กเสริมจากสมการกำลังดัดพลาสติก (Quadratic Equation for Rn)
R_n = M_u_critical / (phi_bending * L * (d ** 2))
m_factor = fy / (0.85 * fc_prime)
rho = (1 / m_factor) * (1 - math.sqrt(1 - (2 * m_factor * R_n / fy))) if (1 - (2 * m_factor * R_n / fy)) > 0 else 0.002

# ตรวจสอบปริมาณเหล็กต่ำสุด
rho_min = 0.002 # สำหรับเหล็กข้ออ้อยชั้นคุณภาพสูง
if rho < rho_min: rho = rho_min

As_total = rho * L * d

# เลือกขนาดและจัดเหล็กเสริม
db_size = 16 # บังคับใช้ขั้นต่ำ DB16 สำหรับงานระดับมาตรฐานสูง
as_single_bar = (math.pi / 4) * (db_size / 10) ** 2
rebar_count = math.ceil(As_total / as_single_bar)
if rebar_count < 5: rebar_count = 5
spacing = ((L - 15) / (rebar_count - 1)) # หักระยะหุ้มซ้ายขวาฝั่งละ 7.5 ซม.

# ==========================================
# VISUALIZATION & REPORT INTERFACE
# ==========================================
st.markdown("---")
t1, t2 = st.tabs(["📋 รายการคำนวณเชิงวิศวกรรม (Calculation Sheet)", "🖼️ แผนผังจำลองโครงสร้างแบบ Real-Time"])

with t1:
    st.subheader("1. ตรวจสอบกำลังรับน้ำหนักของดิน (Geotechnical Bearing Capacity)")
    col_g1, col_g2, col_g3 = st.columns(3)
    
    col_g1.metric("แรงดันดินสูงสุด (q_max)", f"{q_max_tsm:.2f} ตัน/ม²")
    col_g2.metric("แรงดันดินต่ำสุด (q_min)", f"{q_min_tsm:.2f} ตัน/ม²")
    
    if q_max_tsm <= qa_tsm and q_min_tsm >= 0:
        col_g3.success("✅ ผ่าน: ดินรับน้ำหนักได้ และไม่เกิดแรงดึงใต้ฐานราก")
    elif q_min_tsm < 0:
        col_g3.warning("⚠️ เตือน: เกิด Soil Tension (ฐานรากเริ่มเผยอ ต้องขยายขนาดความกว้าง)")
    else:
        col_g3.error("❌ วิกฤต: แรงดันดินเกินความสามารถของชั้นดิน (Soil Failure)")

    st.markdown("---")
    st.subheader("2. ตรวจสอบความปลอดภัยทางโครงสร้าง (Structural Shear Safety - SDM)")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write("**แรงเฉือนคานกว้าง (Wide-Beam Shear):**")
        st.write(f"- หน่วยแรงเฉือนที่เกิดขึ้น ($v_u$): `{v_u_wide:.2f}` ksc")
        st.write(f"- หน่วยแรงเฉือนที่ยอมให้ ($\phi v_c$): `{v_c_wide:.2f}` ksc")
        if v_u_wide <= v_c_wide:
            st.success("✔️ ผ่านความปลอดภัยจากแรงเฉือนคานกว้าง")
        else:
            st.error("❌ หนาไม่พอ: วิกฤตแรงเฉือนคานกว้าง (เพิ่มความหนา H)")
            
    with col_s2:
        st.write("**แรงเฉือนทะลุ (Punching Shear):**")
        st.write(f"- หน่วยแรงเฉือนที่เกิดขึ้น ($v_u$): `{v_u_punch:.2f}` ksc")
        st.write(f"- หน่วยแรงเฉือนที่ยอมให้ ($\phi v_c$): `{v_c_punch:.2f}` ksc")
        if v_u_punch <= v_c_punch:
            st.success("✔️ ผ่านความปลอดภัยจากแรงเฉือนทะลุ")
        else:
            st.error("❌ หนาไม่พอ: วิกฤตแรงเฉือนทะลุเสาตอม่อ (เพิ่มความหนา H)")

    st.markdown("---")
    st.subheader("3. ข้อกำหนดการเสริมเหล็ก (Flexural Reinforcement Summary)")
    st.info(f"📌 **ข้อแนะนำการจัดเหล็กตะแกรงล่าง:** ใช้เหล็กข้ออ้อย **DB{db_size} จำนวน {rebar_count} เส้น** จัดระยะห่างเฉลี่ย **@{spacing:.1f} ซม.** (ใส่เหมือนกันทั้งด้านกว้างและยาว)")

with t2:
    st.subheader("📐 แผนผังจำลองการจัดเหล็กและกระจายแรงดันดิน (Top View Sketch)")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # ฐานราก
    ax.add_patch(plt.Rectangle((0, 0), B_input, L_input, color='#d9d9d9', ec='black', lw=2, label='Footing'))
    # เสาตอม่อ
    cx = (B_input - (col_b/100)) / 2
    cy = (L_input - (col_t/100)) / 2
    ax.add_patch(plt.Rectangle((cx, cy), col_b/100, col_t/100, color='#ff9999', ec='red', lw=1.5))
    
    # วาดเหล็กปลอกตะแกรงคร่าวๆ
    for i in range(rebar_count):
        pos = 0.075 + i * (spacing / 100)
        if pos < L_input:
            ax.plot([0.075, B_input-0.075], [pos, pos], color='blue', lw=1, ls='--')
            ax.plot([pos, pos], [0.075, L_input-0.075], color='blue', lw=1, ls='--')

    ax.set_xlim(-0.2, B_input + 0.2)
    ax.set_ylim(-0.2, L_input + 0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # ใส่ Dimension Text
    ax.text(B_input/2, L_input + 0.05, f"L = {L_input:.2f} m", ha='center', va='bottom', weight='bold')
    ax.text(B_input + 0.05, L_input/2, f"B = {B_input:.2f} m", ha='left', va='center', weight='bold', rotation=-90)
    ax.text(B_input/2, L_input/2, f"Column\n{int(col_b)}x{int(col_t)} cm", ha='center', va='center', color='darkred', fontsize=9)
    
    st.pyplot(fig)
