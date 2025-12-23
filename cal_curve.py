import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="ATPO Real Cal Check", layout="wide")
st.title("🧪 Kiểm tra Cal ATPO (Dữ liệu thực tế)")

# --- 1. THÔNG SỐ MASTER CURVE (TỪ XML) ---
# Lot: 882670
A_master = 876721.0   # Max Signal (Dose 0)
B_master = 0.762881   # Slope
C_master = 175.289    # IC50
D_master = -1315.11   # Min Signal (Infinite Dose)

def get_master_signal(conc):
    """Tính tín hiệu lý thuyết trên đường Master"""
    if conc < 0: return A_master
    return D_master + (A_master - D_master) / (1.0 + (conc / C_master) ** B_master)

def get_concentration(signal, slope, intercept):
    """Tính nồng độ mẫu bệnh nhân từ tín hiệu đo được"""
    # 1. Chuẩn hóa tín hiệu về thang đo Master
    # Meas = Slope * Master + Int => Master = (Meas - Int) / Slope
    sig_norm = (signal - intercept) / slope
    
    # 2. Giải phương trình 4PL ngược
    # y = D + (A-D)/(1+(x/C)^B) => x = C * ((A-D)/(y-D) - 1)^(1/B)
    try:
        term1 = A_master - D_master
        term2 = sig_norm - D_master
        if term2 == 0: return np.nan
        ratio = term1 / term2 - 1
        if ratio <= 0: return np.nan
        return C_master * (ratio ** (1/B_master))
    except:
        return np.nan

# --- 2. GIAO DIỆN NHẬP KẾT QUẢ CAL (TỪ ẢNH) ---
st.subheader("1. Dữ liệu Calibration (Từ màn hình Cobas)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Level 1 (Cal 1)")
    # Giá trị mặc định lấy từ ảnh của bạn
    c1_target = st.number_input("Target 1 (IU/mL)", value=42.1)
    c1_meas_1 = st.number_input("Signal 1 (Lần 1)", value=583602.0)
    c1_meas_2 = st.number_input("Signal 1 (Lần 2)", value=583843.0)
    c1_avg = (c1_meas_1 + c1_meas_2) / 2
    st.info(f"👉 Trung bình Signal 1: **{c1_avg:,.1f}**")

with col2:
    st.markdown("### Level 2 (Cal 2)")
    # Giá trị mặc định lấy từ ảnh của bạn
    c2_target = st.number_input("Target 2 (IU/mL)", value=372.0)
    c2_meas_1 = st.number_input("Signal 2 (Lần 1)", value=289073.0)
    c2_meas_2 = st.number_input("Signal 2 (Lần 2)", value=287568.0)
    c2_avg = (c2_meas_1 + c2_meas_2) / 2
    st.info(f"👉 Trung bình Signal 2: **{c2_avg:,.1f}**")

# --- 3. TÍNH TOÁN & SO SÁNH ---
if st.button("🚀 Thực hiện Recalibration", type="primary"):
    
    # A. Tính tín hiệu Master lý thuyết
    m_sig_1 = get_master_signal(c1_target)
    m_sig_2 = get_master_signal(c2_target)
    
    # B. Tính Slope & Intercept
    # Hệ phương trình tuyến tính đi qua 2 điểm: (Master1, Meas1) và (Master2, Meas2)
    slope = (c2_avg - c1_avg) / (m_sig_2 - m_sig_1)
    intercept = c1_avg - slope * m_sig_1
    
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.subheader("Kết quả Tính toán")
        st.write("Thông số hiệu chuẩn (Calibration Factors):")
        st.metric("Slope (Độ dốc)", f"{slope:.4f}")
        st.metric("Intercept (Chặn)", f"{intercept:,.2f}")
        
        # Đánh giá (Tiêu chuẩn Roche thường là 0.8 - 1.2)
        if 0.8 <= slope <= 1.2:
            st.success("✅ CAL PASSED (Đạt chuẩn)")
        else:
            st.error("❌ CAL FAILED (Ngoài dải cho phép)")
            
        st.markdown("---")
        st.markdown("**Giải thích:**")
        st.caption(f"Tín hiệu Master tại 42.1 IU/mL: {m_sig_1:,.0f}")
        st.caption(f"Tín hiệu Master tại 372 IU/mL: {m_sig_2:,.0f}")
        st.caption(f"Máy đang hoạt động ở mức **{slope*100:.1f}%** tín hiệu so với lúc xuất xưởng.")

    with res_col2:
        st.subheader("Biểu đồ Đường chuẩn")
        
        # Vẽ đường cong
        x_plot = np.logspace(np.log10(5), np.log10(1000), 200)
        
        # 1. Đường Master Gốc
        y_master = [get_master_signal(x) for x in x_plot]
        
        # 2. Đường Thực tế (Recalibrated)
        y_recal = [val * slope + intercept for val in y_master]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_plot, y=y_master, mode='lines', name='Master Curve (Nhà máy)', line=dict(dash='dash', color='gray')))
        fig.add_trace(go.Scatter(x=x_plot, y=y_recal, mode='lines', name='Actual Curve (Hôm nay)', line=dict(color='blue')))
        
        # Điểm Cal
        fig.add_trace(go.Scatter(
            x=[c1_target, c2_target], y=[c1_avg, c2_avg],
            mode='markers', name='Điểm Cal Lab', marker=dict(size=12, color='red', symbol='cross')
        ))

        fig.update_layout(
            xaxis_type="log", yaxis_type="log",
            xaxis_title="Nồng độ ATPO (IU/mL)",
            yaxis_title="Tín hiệu (Counts)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 4. TÍNH MẪU THỬ ---
    st.divider()
    st.subheader("🧪 Thử tính mẫu bệnh nhân")
    c_test_sig = st.number_input("Nhập Tín hiệu mẫu (Ví dụ: 400000)", value=400000.0)
    
    if st.button("Tính kết quả mẫu"):
        res = get_concentration(c_test_sig, slope, intercept)
        st.success(f"Kết quả nồng độ: **{res:.4f} IU/mL**")
        
        # Vẽ điểm này lên đồ thị
        fig.add_trace(go.Scatter(
            x=[res], y=[c_test_sig],
            mode='markers', name='Mẫu Bệnh Nhân', marker=dict(size=15, color='green', symbol='star')
        ))
        with res_col2:
            st.plotly_chart(fig, use_container_width=True, key="update_chart")
