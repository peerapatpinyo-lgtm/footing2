import streamlit as st
import math
import matplotlib.pyplot as plt

# ตั้งค่าหน้าเว็บให้กว้างและสวยงาม
st.set_page_config(page_title="Pro Footing Designer (WSD)", layout="wide")

st.title("🏗️ Pro Isolated Footing Designer (วิธี WSD)")
st.caption("พัฒนาตามมาตรฐาน วสท. | คำนวณโครงสร้างพร้อมวาดแบบขยายเรียลไทม์")
st.markdown("---")

# ==========================================
# ส่วนรับข้อมูลอินพุต (Sidebar)
# ==========================================
st.sidebar.header("📥 พารามิเตอร์การออกแบบ")

with st.sidebar.expander("💥 น้ำหนักบรรทุก & ดิน", expanded=True):
    P_tons = st.number_input("น้ำหนักบรรทุกใช้งาน P (ตัน)", min_value=1.0, value=35.0, step=1.0)
    qa_tsm = st.number_input("กำลังรับน้ำหนักปลอดภัยของดิน qa (ตัน/ตร.ม.)", min_value=1.0, value=15.0, step=0.5)

with st.sidebar.expander("🧱 วัสดุ (Concrete & Steel)", expanded=True):
    fc_prime = st.number_input("กำลังอัดคอนกรีต fc' (ksc)", min_value=100, value=240, step=10)
    fy = st.selectbox("จุดคราดเหล็กเสริม fy (ksc)", options=[2400, 3000, 4000], index=2)

with st.sidebar.expander("📐 ขนาดเสาตอม่อ (ซม.)", expanded=True):
    col_b = st.number_input("ความกว้างเสา (b)", min_value=10.0, value=20.0, step=5.0)
    col_t = st.number_input("ความลึกเสา (t)", min_value=10.0, value=20.0, step=5.0)

# ==========================================
# กระบวนการคำนวณ (Engine)
# ==========================================
# แปลงหน่วย
P = P_tons * 1000  
qa = qa_tsm / 10   

# ค่าคงที่ WSD
fc = 0.375 * fc_prime
fs = 1500 if fy < 3000 else 1700
n = 135 / math.sqrt(fc_prime)
k = 1 / (1 + (fs / (n * fc)))
j = 1 - (k / 3)

v_wide_allow = 0.29 * math.sqrt(fc_prime)
v_punch_allow = 0.53 * math.sqrt(fc_prime)

# หาขนาดฐานราก (เผื่อน้ำหนักตัวเอง 10%)
P_total = P * 1.10
area_req = P_total / qa
B_exact = math.sqrt(area_req)
B = math.ceil(B_exact / 10) * 10  # ปัดขึ้นทีละ 10 ซม.

q_net = P / (B * B)

# ลูปหาความหนา d เพื่อความปลอดภัยจากแรงเฉือน
d = 15.0
covering = 7.5

while True:
    x = (B - col_b) / 2
    
    # แรงเฉือนคานกว้าง
    V_wide = q_net * B * (x - d)
    v_wide = V_wide / (B * d)
    
    # แรงเฉือนทะลุ
    bo = 2 * ((col_b + d) + (col_t + d))
    area_punch = (col_b + d) * (col_t + d)
    V_punch = q_net * ((B * B) - area_punch)
    v_punch = V_punch / (bo * d)
    
    if v_wide <= v_wide_allow and v_punch <= v_punch_allow:
        break
    d += 1.0

total_depth = d + covering

# คำนวณโมเมนต์และเหล็กเสริม
M = (q_net * B * (x ** 2)) / 2
As_req = M / (fs * j * d)
As_min = 0.002 * B * total_depth
As_final = max(As_req, As_min)

# เลือกขนาดเหล็กเสริมอัตโนมัติ
db_dia = 1.2 if B < 150 else 1.6
as_bar = (math.pi / 4) * (db_dia ** 2)
num_bars = math.ceil(As_final / as_bar)
if num_bars < 4: num_bars = 4 # ขั้นต่ำ 4 เส้น
spacing = (B - (2 * covering)) / (num_bars - 1)

# ==========================================
# ส่วนแสดงผลลัพธ์ (UI Main Area)
# ==========================================
# แบ่งหน้าตาโปรแกรมเป็น 2 แท็บ
tab1, tab2 = st.tabs(["📊 ผลการคำนวณอย่างละเอียด", "📐 แบบขยายโครงสร้าง (2D Drawing)"])

with tab1:
    st.subheader("💡 สรุปขนาดฐานรากที่เหมาะสม")
    
    # แสดงเมทริกซ์หลัก
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("กว้าง x ยาว (ม.)", f"{B/100:.2f} x {B/100:.2f}")
    m2.metric("ความหนา H (ซม.)", f"{total_depth:.1f}")
    m3.metric("น้ำหนักบรรทุกเฉลี่ย (ตัน/ม²)", f"{q_net*10:.2f}")
    m4.metric("เหล็กเสริมที่ใช้", f"DB{int(db_dia*10)} @ {spacing:.1f} ซม.")

    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 🛡️ การตรวจสอบทางวิศวกรรม")
        
        # กล่องตรวจสอบแรงเฉือนคานกว้าง
        if v_wide <= v_wide_allow:
            st.success(f"🔹 **แรงเฉือนคานกว้าง (ผ่าน):** {v_wide:.2f} ksc (หน่วยแรงที่ยอมให้ {v_wide_allow:.2f} ksc)")
        else:
            st.error(f"🔺 **แรงเฉือนคานกว้าง (ไม่ผ่าน):** {v_wide:.2f} ksc")
            
        # กล่องตรวจสอบแรงเฉือนทะลุ
        if v_punch <= v_punch_allow:
            st.success(f"🔹 **แรงเฉือนทะลุ (ผ่าน):** {v_punch:.2f} ksc (หน่วยแรงที่ยอมให้ {v_punch_allow:.2f} ksc)")
        else:
            st.error(f"🔺 **แรงเฉือนทะลุ (ไม่ผ่าน):** {v_punch:.2f} ksc")
            
    with col_right:
        st.markdown("### 🧲 ปริมาณเหล็กเสริมตามคำนวณ")
        st.write(f"• โมเมนต์ดัดวิกฤต ($M_x$): `{M/100000:.2f}` ตัน-เมตร")
        st.write(f"• เนื้อที่เหล็กเสริมต้องการตามคำนวณ: `{As_req:.2f}` ตร.ซม.")
        st.write(f"• เนื้อที่เหล็กเสริมขั้นต่ำตามข้อกำหนด (0.002bt): `{As_min:.2f}` ตร.ซม.")
        st.write(f"📌 **เลือกใช้เนื้อที่เหล็กเสริม:** `{As_final:.2f}` ตร.ซม.")

with tab2:
    st.subheader("🛠️ แบบขยายการเสริมเหล็กฐานรากแผ่ (Top View)")
    st.caption("หมายเหตุ: ระยะต่างๆ มีหน่วยเป็นเซนติเมตร (cm)")
    
    # สร้างรูปภาพด้วย Matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. วาดตัวฐานราก (สี่เหลี่ยมด้านนอก)
    rect_footing = plt.Rectangle((0, 0), B, B, linewidth=2, edgecolor='black', facecolor='#f0f0f0', label='Footing')
    ax.add_patch(rect_footing)
    
    # 2. วาดเสาตอม่อตรงกลาง
    col_x = (B - col_b) / 2
    col_y = (B - col_t) / 2
    rect_column = plt.Rectangle((col_x, col_y), col_b, col_t, linewidth=2, edgecolor='darkred', facecolor='#ffcccc', label='Column')
    ax.add_patch(rect_column)
    
    # 3. วาดเหล็กเสริม (เส้นแกน X และ แกน Y)
    # เหล็กเส้นแนวนอน
    for i in range(num_bars):
        y_pos = covering + (i * spacing)
        ax.plot([covering, B - covering], [y_pos, y_pos], color='blue', linewidth=1.5, linestyle='--')
        
    # เหล็กเส้นแนวตั้ง
    for i in range(num_bars):
        x_pos = covering + (i * spacing)
        ax.plot([x_pos, x_pos], [covering, B - covering], color='darkblue', linewidth=1.5, linestyle='--')
        
    # ปรับแต่งการแสดงผลของกราฟ
    ax.set_xlim(-10, B + 10)
    ax.set_ylim(-10, B + 10)
    ax.set_aspect('equal')
    ax.axis('off') # ปิดแกนตัวเลขไม้บรรทัดกราฟเพื่อความสวยงามเหมือนแบบสถาปัตย์
    
    # ใส่ข้อความกำกับขนาด
    ax.text(B/2, B + 2, f"B = {B:.0f} cm", ha='center', va='bottom', fontsize=12, weight='bold')
    ax.text(B + 2, B/2, f"B = {B:.0f} cm", ha='left', va='center', fontsize=12, weight='bold', rotation=-90)
    ax.text(B/2, B/2, f"Col {col_b:.0f}x{col_t:.0f}", ha='center', va='center', color='darkred', weight='bold')
    
    # แสดงคำอธิบายสัญลักษณ์
    st.pyplot(fig)
    
    # ข้อความสรุปใต้ภาพ
    st.info(f"📋 **สรุปรายการเหล็กเสริมสำหรับช่าง:** ตะแกรงล่าง ใช้เหล็ก **DB{int(db_dia*10)} จำนวน {num_bars} เส้น** จัดห่างกันทุกๆ **@{spacing:.1f} ซม.** (ทั้งสองทิศทาง)")
