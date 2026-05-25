import streamlit as st
import math

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="โปรแกรมออกแบบฐานรากแผ่ (WSD)", layout="centered")

st.title("🏗️ โปรแกรมออกแบบฐานรากแผ่เดี่ยว (วิธี WSD)")
st.write("อ้างอิงตามมาตรฐานวิศวกรรมสถานแห่งประเทศไทย (วสท.)")
st.markdown("---")

# ส่วนรับข้อมูลอินพุต (Sidebar)
st.sidebar.header("📥 ข้อมูลสำหรับออกแบบ")

P_tons = st.sidebar.number_input("น้ำหนักบรรทุกใช้งาน P (ตัน)", min_value=1.0, value=35.0, step=1.0)
qa_tsm = st.sidebar.number_input("กำลังรับน้ำหนักปลอดภัยของดิน qa (ตัน/ตร.ม.)", min_value=1.0, value=15.0, step=0.5)
fc_prime = st.sidebar.number_input("กำลังอัดประลัยของคอนกรีต fc' (ksc)", min_value=100, value=240, step=10)
fy = st.sidebar.selectbox("จุดคราดของเหล็กเสริม fy (ksc)", options=[2400, 3000, 4000], index=2)

st.sidebar.subheader("ขนาดเสาตอม่อ (ซม.)")
col_b_cm = st.sidebar.number_input("ความกว้างเสา (ซม.)", min_value=10.0, value=20.0, step=5.0)
col_t_cm = st.sidebar.number_input("ความลึกเสา (ซม.)", min_value=10.0, value=20.0, step=5.0)

# ปุ่มคำนวณ
if st.sidebar.button("🚀 คำนวณออกแบบฐานราก"):
    
    # --- เริ่มกระบวนการคำนวณ ---
    P = P_tons * 1000  # kg
    qa = qa_tsm / 10   # kg/cm^2
    col_b = col_b_cm
    col_t = col_t_cm
    
    # กำหนดค่าหน่วยแรงที่ยอมให้ (WSD)
    fc = 0.375 * fc_prime
    fs = 1500 if fy < 3000 else 1700
    n = 135 / math.sqrt(fc_prime)
    k = 1 / (1 + (fs / (n * fc)))
    j = 1 - (k / 3)
    
    v_wide_allow = 0.29 * math.sqrt(fc_prime)
    v_punch_allow = 0.53 * math.sqrt(fc_prime)
    
    # หาขนาดฐานราก (เผื่อน้ำหนักฐานราก 10%)
    P_total = P * 1.10
    area_req = P_total / qa
    B_exact = math.sqrt(area_req)
    B = math.ceil(B_exact / 10) * 10  # ปัดขึ้นทีละ 10 ซม.
    
    q_net = P / (B * B)
    
    # ลูปหาความหนา d
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
    
    # คำนวณระยะห่างเหล็กเสริม
    db_dia = 1.2 if B < 150 else 1.6
    as_bar = (math.pi / 4) * (db_dia ** 2)
    num_bars = math.ceil(As_final / as_bar)
    spacing = (B - (2 * covering)) / (num_bars - 1)
    
    # --- แสดงผลลัพธ์บน Streamlit ---
    st.subheader("📊 ผลการตรวจสอบและขนาดฐานราก")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("ขนาดฐานราก (B x B)", f"{B/100:.2f} x {B/100:.2f} ม.")
    col2.metric("ความหนาทั้งหมด (H)", f"{total_depth:.1f} ซม.")
    col3.metric("แรงดันดินจริง (q_net)", f"{q_net*10:.2f} ตัน/ตร.ม.")
    
    st.markdown("### 🛡️ การตรวจสอบแรงเฉือน")
    
    # แสดงสถานะแรงเฉือนคานกว้าง
    if v_wide <= v_wide_allow:
        st.success(f"✔️ **แรงเฉือนคานกว้าง ผ่าน:** {v_wide:.2f} ksc (ยอมให้ {v_wide_allow:.2f} ksc)")
    else:
        st.error(f"❌ **แรงเฉือนคานกว้าง ไม่ผ่าน:** {v_wide:.2f} ksc")
        
    # แสดงสถานะแรงเฉือนทะลุ
    if v_punch <= v_punch_allow:
        st.success(f"✔️ **แรงเฉือนทะลุ ผ่าน:** {v_punch:.2f} ksc (ยอมให้ {v_punch_allow:.2f} ksc)")
    else:
        st.error(f"❌ **แรงเฉือนทะลุ ไม่ผ่าน:** {v_punch:.2f} ksc")
        
    st.markdown("### 🧲 การเสริมเหล็ก (Reinforcement)")
    st.write(f"• โมเมนต์ดัดวิกฤตที่ขอบเสา: **{M/100000:.2f} ตัน-เมตร**")
    st.write(f"• ปริมาณเหล็กเสริมที่ต้องการ: **{As_final:.2f} ตร.ซม.**")
    
    st.info(f"💡 **ข้อเสนอแนะการจัดเหล็ก:** จัดเหล็ก **DB{int(db_dia*10)}** จำนวน **{num_bars} เส้น** "
            f"ในแต่ละทิศทาง (ระยะห่างเฉลี่ย @ **{spacing:.1f} ซม.**)")
else:
    st.info("👈 กรุณากรอกข้อมูลที่แถบด้านซ้าย แล้วกดปุ่ม **'คำนวณออกแบบฐานราก'** เพื่อดูผลลัพธ์")
