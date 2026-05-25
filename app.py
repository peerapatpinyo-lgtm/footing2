import math

def design_isolated_footing(P_tons, qa_tsm, fc_prime, fy, col_b_cm, col_t_cm):
    """
    โปรแกรมออกแบบฐานรากแผ่เดี่ยว (วิธี WSD) ตามมาตรฐาน วสท.
    P_tons: น้ำหนักบรรทุกใช้งาน (ตัน)
    qa_tsm: กำลังรับน้ำหนักปลอดภัยของดิน (ตัน/ตร.ม.)
    fc_prime: กำลังอัดประลัยของคอนกรีตที่ 28 วัน (ksc)
    fy: จุดคราดของเหล็กเสริม (ksc)
    col_b_cm, col_t_cm: ขนาดเสา กว้าง x ยาว (ซม.)
    """
    
    print("--- ผลการคำนวณออกแบบฐานรากแผ่ ---")
    
    # 1. แปลงหน่วยเป็น kg และ cm
    P = P_tons * 1000  # kg
    qa = qa_tsm / 10   # kg/cm^2
    col_b = col_b_cm
    col_t = col_t_cm
    
    # กำหนดค่าหน่วยแรงที่ยอมให้อ้างอิง วสท.
    fc = 0.375 * fc_prime
    fs = 1500 if fy < 3000 else 1700  # SR24 หรือ SD30/40
    n = 135 / math.sqrt(fc_prime)
    k = 1 / (1 + (fs / (n * fc)))
    j = 1 - (k / 3)
    R = 0.5 * fc * k * j
    
    # หน่วยแรงเฉือนที่ยอมให้ของคอนกรีต (WSD)
    v_wide_allow = 0.29 * math.sqrt(fc_prime)
    v_punch_allow = 0.53 * math.sqrt(fc_prime)
    
    # 2. หาขนาดฐานราก (สมมติน้ำหนักฐานรากเพิ่ม 10%)
    P_total = P * 1.10
    area_req = P_total / qa
    B_exact = math.sqrt(area_req)
    
    # ปัดขนาดฐานรากขึ้นทีละ 10 ซม. เพื่อความง่ายในการก่อสร้าง
    B = math.ceil(B_exact / 10) * 10
    print(f"ขนาดฐานรากที่ใช้: {B/100:.2f} x {B/100:.2f} ม.")
    
    # แรงดันดินใช้งานจริง (Net Upward Pressure)
    q_net = P / (B * B)
    
    # 3. ลูปหาความหนาแน่น (d) ที่ปลอดภัยจากแรงเฉือน
    d = 15.0  # เริ่มต้นที่ความหนาประสิทธิผล 15 ซม.
    covering = 7.5
    
    while True:
        # ระยะยื่นจากขอบเสา
        x = (B - col_b) / 2
        
        # --- ตรวจสอบแรงเฉือนคานกว้าง (Wide-Beam Shear) ที่ระยะ d ---
        V_wide = q_net * B * (x - d)
        v_wide = V_wide / (B * d)
        
        # --- ตรวจสอบแรงเฉือนทะลุ (Punching Shear) ที่ระยะ d/2 ---
        bo = 2 * ((col_b + d) + (col_t + d))  # เส้นรอบรูปวิกฤต
        area_punch = (col_b + d) * (col_t + d)
        V_punch = q_net * ((B * B) - area_punch)
        v_punch = V_punch / (bo * d)
        
        if v_wide <= v_wide_allow and v_punch <= v_punch_allow:
            break
        d += 1.0  # ถ้าไม่ผ่าน ให้เพิ่มความหนาทีละ 1 ซม.
        
    total_depth = d + covering
    print(f"ความหนาประสิทธิผล (d): {d:.1f} ซม.")
    print(f"ความหนาฐานรากทั้งหมด (H): {total_depth:.1f} ซม. (Covering 7.5 cm)")
    print(f"แรงเฉือนคานกว้าง: {v_wide:.2f} / ยอมให้ {v_wide_allow:.2f} ksc -> {'ผ่าน' if v_wide <= v_wide_allow else 'ไม่ผ่าน'}")
    print(f"แรงเฉือนทะลุ: {v_punch:.2f} / ยอมให้ {v_punch_allow:.2f} ksc -> {'ผ่าน' if v_punch <= v_punch_allow else 'ไม่ผ่าน'}")
    
    # 4. คำนวณโมเมนต์ดัดและเหล็กเสริม (Bending Moment & Reinforcement)
    # คิดโมเมนต์ที่ขอบเสา
    M = (q_net * B * (x ** 2)) / 2  # kg-cm
    
    # หาเนื้อที่เหล็กเสริม
    As_req = M / (fs * j * d)
    
    # ตรวจสอบเหล็กเสริมขั้นต่ำ (Minimum Reinforcement อิงตามอัตราส่วน 0.002 สำหรับเหล็กข้ออ้อย)
    As_min = 0.002 * B * total_depth
    As_final = max(As_req, As_min)
    
    print(f"โมเมนต์ดัดวิกฤต: {M/100000:.2f} ตัน-เมตร")
    print(f"ปริมาณเหล็กเสริมที่ต้องการทั้งหมดในหนึ่งทิศทาง: {As_final:.2f} ตร.ซม.")
    
    # แนะนำการจัดเหล็กเสริมเบื้องต้น (เช่น ใช้เหล็ก DB12 หรือ DB16)
    db_dia = 1.2 if B < 150 else 1.6 # เลือก DB12 สำหรับฐานรากเล็ก DB16 สำหรับฐานรากใหญ่
    as_bar = (math.pi / 4) * (db_dia ** 2)
    num_bars = math.ceil(As_final / as_bar)
    spacing = (B - (2 * covering)) / (num_bars - 1)
    
    print(f"คำแนะนำการเสริมเหล็ก: DB{int(db_dia*10)} จำนวน {num_bars} เส้น (ระยะห่างประมาณ @ {spacing:.1f} ซม.)")
    print("--------------------------------")

# --- ทดลองรันโปรแกรมด้วยค่าตัวอย่าง ---
# น้ำหนักบรรทุก 35 ตัน, ดินรับได้ 15 ตัน/ตร.ม., คอนกรีต fc'=240 ksc, เหล็ก SD40 (fy=4000 ksc), เสาขนาด 20x20 ซม.
design_isolated_footing(P_tons=35, qa_tsm=15, fc_prime=240, fy=4000, col_b_cm=20, col_t_cm=20)
