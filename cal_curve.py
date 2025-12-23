import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="ATPO Real Cal Check", layout="wide")
st.title("🧪 Kiểm tra Cal ATPO (Dữ liệu thực tế)")

# --- 0. KHỞI TẠO SESSION STATE (QUAN TRỌNG) ---
# Kiểm tra xem các biến này đã có trong bộ nhớ chưa, nếu chưa thì tạo mới
if 'A_val' not in st.session_state: st.session_state.A_val = 876721.0
if 'B_val' not in st.session_state: st.session_state.B_val = 0.762881
if 'C_val' not in st.session_state: st.session_state.C_val = 175.289
if 'D_val' not in st.session_state: st.session_state.D_val = -1315.11

# --- 1. NHẬP THAM SỐ MASTER CURVE (CÓ LƯU TRẠNG THÁI) ---
with st.sidebar:
    st.header("Cấu hình Master Curve")
    st.info("Nhập tham số từ XML/Barcode (Sẽ được lưu lại khi bấm Tính)")
    
    # Thay vì dùng biến thường, ta dùng key=... để liên kết với session_state
    A_master = st.number_input("Tham số A (Max)", value=st.session_state.A_val, key='A_input', format="%.2f")
    B_master = st.number_input("Tham số B (Slope)", value=st.session_state.B_val, key='B_input', format="%.6f")
    C_master = st.number_input("Tham số C (IC50)", value=st.session_state.C_val, key='C_input', format="%.4f")
    D_master = st.number_input("Tham số D (Min)", value=st.session_state.D_val, key='D_input', format="%.2f")
    
    # Cập nhật ngược lại vào session_state (để chắc chắn)
    st.session_state.A_val = A_master
    st.session_state.B_val = B_master
    st.session_state.C_val = C_master
    st.session_state.D_val = D_master

# --- HÀM TOÁN HỌC ---
def get_master_signal(conc):
    """Tính tín hiệu lý thuyết trên đường Master"""
    if conc < 0: return A_master
    # Sử dụng trực tiếp biến A_master, B_master... vừa lấy từ input
    return D_master + (A_master - D_master) / (1.0 + (conc / C_master) ** B_master)

def get_concentration(signal, slope, intercept):
    """Tính nồng độ mẫu bệnh nhân"""
    sig_norm = (signal - intercept) / slope
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
    c1_target = st.number_input("Target 1 (IU/mL)", value=42.1)
    c1_meas_1 = st.number_input("Signal 1 (Lần 1)", value=583602.0)
    c1_meas_2 = st.number_input("Signal 1 (Lần 2)", value=583843.0)
    c1_avg = (c1_meas_1 + c1_meas_2) / 2
    st.info(f"👉 Trung bình Signal 1: **{c1_avg:,.1f}**")

with col2:
    st.markdown("### Level 2 (Cal 2)")
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
    slope = (c2_avg - c1_avg) / (m_sig_2 - m_sig_1)
    intercept = c1_avg - slope * m_sig_1
    
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.subheader("Kết quả Tính toán")
        st.write("Thông số hiệu chuẩn (Calibration Factors):")
        st.metric("Slope (Độ dốc)", f"{slope:.4f}")
        st.metric("Intercept (Chặn)", f"{intercept:,.2f}")
        
        if 0.8 <= slope <= 1.2:
            st.success("✅ CAL PASSED (Đạt chuẩn)")
        else:
            st.error("❌ CAL FAILED (Ngoài dải cho phép)")
            
        st.caption(f"Tín hiệu Master tại {c1_target}: {m_sig_1:,.0f}")
        st.caption(f"Tín hiệu Master tại {c2_target}: {m_sig_2:,.0f}")

    with res_col2:
        st.subheader("Biểu đồ Đường chuẩn")
        
        x_plot = np.logspace(np.log10(5), np.log10(1000), 200)
        y_master = [get_master_signal(x) for x in x_plot]
        y_recal = [val * slope + intercept for val in y_master]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_plot, y=y_master, mode='lines', name='Master Curve (Gốc)', line=dict(dash='dash', color='gray')))
        fig.add_trace(go.Scatter(x=x_plot, y=y_recal, mode='lines', name='Actual Curve (Hiện tại)', line=dict(color='blue')))
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

    # --- 4. TÍNH MẪU THỬ (Lồng bên trong để dùng biến slope/intercept) ---
    st.divider()
    st.subheader("🧪 Thử tính mẫu bệnh nhân")
    
    # Form giúp gom nhóm nhập liệu để trông gọn hơn
    with st.form("sample_calc_form"):
        c_test_sig = st.number_input("Nhập Tín hiệu mẫu (Ví dụ: 400000)", value=400000.0)
        calc_submitted = st.form_submit_button("Tính kết quả mẫu")
        
        if calc_submitted:
            res = get_concentration(c_test_sig, slope, intercept)
            st.success(f"Kết quả nồng độ: **{res:.4f} IU/mL**")
            
            # Vẽ thêm điểm này
            fig.add_trace(go.Scatter(
                x=[res], y=[c_test_sig],
                mode='markers', name='Mẫu Bệnh Nhân', marker=dict(size=15, color='green', symbol='star')
            ))
            # Cần vẽ lại chart để hiện điểm mới
            st.plotly_chart(fig, use_container_width=True, key="chart_updated")
