import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Roche Cal Troubleshoot", layout="wide")
st.title("📈 Roche Calibration Troubleshoot & Trend Analysis")
st.markdown("Công cụ theo dõi lịch sử đường chuẩn, phát hiện xu hướng trôi (Drift) và đánh giá độ ổn định của hệ thống.")

# --- 1. KHỞI TẠO SESSION STATE ---
# Lưu tham số Master Curve
if 'master_params' not in st.session_state:
    # Mặc định theo ví dụ ATPO cũ của bạn
    st.session_state.master_params = {
        'A': 876721.0, 'B': 0.762881, 'C': 175.289, 'D': -1315.11
    }

# --- 2. HÀM TOÁN HỌC ---
def get_master_signal(conc, A, B, C, D):
    if conc < 0: return A
    return D + (A - D) / (1.0 + (conc / C) ** B)

# --- 3. SIDEBAR: THAM SỐ MASTER CURVE ---
with st.sidebar:
    st.header("1. Master Curve (Cố định)")
    st.caption("Thông số từ XML/Barcode của Lô thuốc thử đang dùng.")
    
    m_A = st.number_input("A (Max/Dose 0)", value=st.session_state.master_params['A'], format="%.0f")
    m_B = st.number_input("B (Slope)", value=st.session_state.master_params['B'], format="%.6f")
    m_C = st.number_input("C (IC50)", value=st.session_state.master_params['C'], format="%.4f")
    m_D = st.number_input("D (Min/Inf)", value=st.session_state.master_params['D'], format="%.0f")
    
    # Cập nhật lại session state nếu người dùng sửa
    st.session_state.master_params = {'A': m_A, 'B': m_B, 'C': m_C, 'D': m_D}
    
    st.divider()
    st.info("""
    **Hướng dẫn:**
    1. Nhập tham số Master Curve.
    2. Nhập lịch sử các lần Cal vào bảng bên phải.
    3. Xem biểu đồ để phát hiện bất thường.
    """)

# --- 4. GIAO DIỆN CHÍNH: NHẬP LIỆU HÀNG LOẠT ---
st.subheader("2. Lịch sử Calibration (Data Entry)")

# Tạo dữ liệu mẫu (Giả lập lịch sử Cal trong 1 tuần)
# Logic: Tín hiệu giảm dần theo thời gian (Máy già/Thuốc thử hủy)
default_history = pd.DataFrame([
    {"Date": "2023-12-01", "Target L1": 42.1, "Target L2": 372.0, "Signal L1": 590000, "Signal L2": 295000, "Note": "Mới mở lọ"},
    {"Date": "2023-12-08", "Target L1": 42.1, "Target L2": 372.0, "Signal L1": 585000, "Signal L2": 290000, "Note": ""},
    {"Date": "2023-12-15", "Target L1": 42.1, "Target L2": 372.0, "Signal L1": 583602, "Signal L2": 289073, "Note": "Hiện tại"},
    {"Date": "2023-12-22", "Target L1": 42.1, "Target L2": 372.0, "Signal L1": 550000, "Signal L2": 260000, "Note": "Dự báo lỗi"},
])

# Cho phép người dùng sửa bảng
edited_df = st.data_editor(default_history, num_rows="dynamic", use_container_width=True)

# Nút Phân tích
if st.button("🔍 Phân tích Xu hướng (Analyze)", type="primary"):
    
    # --- 5. XỬ LÝ SỐ LIỆU ---
    results = []
    
    # Lấy tham số Master
    p = st.session_state.master_params
    
    for index, row in edited_df.iterrows():
        try:
            # Lấy dữ liệu dòng
            date = row['Date']
            t1, t2 = float(row['Target L1']), float(row['Target L2'])
            s1, s2 = float(row['Signal L1']), float(row['Signal L2'])
            
            # Tính Master Signal
            m1 = get_master_signal(t1, p['A'], p['B'], p['C'], p['D'])
            m2 = get_master_signal(t2, p['A'], p['B'], p['C'], p['D'])
            
            # Tính Slope & Intercept
            # Slope = (S2 - S1) / (M2 - M1)
            slope = (s2 - s1) / (m2 - m1)
            intercept = s1 - slope * m1
            
            # Đánh giá
            status = "Pass"
            if slope < 0.8 or slope > 1.2: status = "Fail"
            
            results.append({
                "Date": date,
                "Slope": slope,
                "Intercept": intercept,
                "Signal L1": s1,
                "Signal L2": s2,
                "Status": status,
                "Target L1": t1, # Lưu để vẽ
                "Target L2": t2  # Lưu để vẽ
            })
            
        except Exception as e:
            st.warning(f"Lỗi dữ liệu tại dòng {index}: {e}")

    # Chuyển kết quả thành DataFrame
    res_df = pd.DataFrame(results)

    # --- 6. HIỂN THỊ DASHBOARD ---
    st.divider()
    st.header("3. Kết quả Chẩn đoán (Troubleshooting Dashboard)")
    
    # A. THẺ KPI TỔNG QUAN
    kpi1, kpi2, kpi3 = st.columns(3)
    latest = res_df.iloc[-1] # Lấy lần Cal mới nhất
    
    kpi1.metric("Lần Cal mới nhất", f"{latest['Date']}")
    kpi2.metric("Hệ số Slope hiện tại", f"{latest['Slope']:.4f}", 
                delta=f"{latest['Slope'] - 1.0:.2f} so với chuẩn", 
                delta_color="inverse") # Slope càng xa 1 càng tệ
    
    status_color = "normal" if latest['Status'] == "Pass" else "off"
    kpi3.metric("Trạng thái", latest['Status'])

    # B. BIỂU ĐỒ 1: XU HƯỚNG SLOPE (QUAN TRỌNG NHẤT)
    st.subheader("📊 Biểu đồ xu hướng hệ số Slope (Calibration Factor)")
    st.caption("Đây là chỉ số quan trọng nhất. Nếu đường này đi xuống liên tục -> Thuốc thử hỏng hoặc Đèn già.")
    
    fig_trend = go.Figure()
    
    # Vùng an toàn (0.8 - 1.2)
    fig_trend.add_hrect(y0=0.8, y1=1.2, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Vùng An Toàn")
    
    # Đường Slope
    fig_trend.add_trace(go.Scatter(
        x=res_df['Date'], y=res_df['Slope'],
        mode='lines+markers', name='Slope',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))
    
    # Điểm Fail
    fails = res_df[res_df['Status'] == 'Fail']
    if not fails.empty:
        fig_trend.add_trace(go.Scatter(
            x=fails['Date'], y=fails['Slope'],
            mode='markers', name='Failed Cal',
            marker=dict(color='red', size=15, symbol='x')
        ))

    fig_trend.update_layout(yaxis_title="Slope Factor", template="plotly_white", height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

    # C. BIỂU ĐỒ 2: OVERLAY MASTER CURVE
    st.subheader("📉 Kiểm tra độ lệch so với Master Curve")
    col_chart_2, col_advice = st.columns([2, 1])
    
    with col_chart_2:
        # Vẽ Master Curve
        x_draw = np.logspace(np.log10(5), np.log10(1000), 200)
        y_master = [get_master_signal(x, p['A'], p['B'], p['C'], p['D']) for x in x_draw]
        
        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Scatter(x=x_draw, y=y_master, mode='lines', name='Master Curve (Gốc)', line=dict(dash='dash', color='gray')))
        
        # Vẽ các điểm Cal lịch sử
        # Màu đậm nhạt theo thời gian (Cũ = Nhạt, Mới = Đậm)
        for i, row in res_df.iterrows():
            opacity = 0.3 + (0.7 * (i / len(res_df))) # Tăng dần độ đậm
            name = f"Cal {row['Date']}" if i == len(res_df)-1 else None # Chỉ hiện tên cái cuối
            
            fig_overlay.add_trace(go.Scatter(
                x=[row['Target L1'], row['Target L2']],
                y=[row['Signal L1'], row['Signal L2']],
                mode='lines+markers',
                line=dict(color='blue', width=1),
                opacity=opacity,
                showlegend=False
            ))
            
        # Highlight lần mới nhất
        fig_overlay.add_trace(go.Scatter(
            x=[latest['Target L1'], latest['Target L2']],
            y=[latest['Signal L1'], latest['Signal L2']],
            mode='markers', name='Lần Cal Mới Nhất',
            marker=dict(color='red', size=12)
        ))

        fig_overlay.update_layout(xaxis_type="log", yaxis_type="log", title="Độ tản mạn các lần Cal", height=450)
        st.plotly_chart(fig_overlay, use_container_width=True)

    # D. PHẦN CHẨN ĐOÁN (TROUBLESHOOTING ADVICE)
    with col_advice:
        st.info("💡 **Phân tích:**")
        
        # Logic phân tích đơn giản
        slope_change = res_df['Slope'].max() - res_df['Slope'].min()
        latest_slope = latest['Slope']
        
        if latest_slope < 0.8:
            st.error("⛔ **LỖI CALIBRATION!** Slope < 0.8. Tín hiệu quá thấp.")
            st.markdown("""
            *Nguyên nhân khả thi:*
            - Thuốc thử hết hạn hoặc để ngoài quá lâu.
            - Kim hút mẫu bị tắc/nghẹt.
            - Bóng đèn quang kế quá già (kiểm tra Photometer Check).
            """)
        elif latest_slope > 1.2:
            st.error("⛔ **LỖI CALIBRATION!** Slope > 1.2. Tín hiệu quá cao.")
            st.markdown("""
            *Nguyên nhân khả thi:*
            - Nhiễm chéo mẫu (Carry-over).
            - Lỗi pha Calibrator (pha quá đặc).
            - Bọt khí trong cuvet đo.
            """)
        else:
            st.success("✅ **Hệ thống ỔN ĐỊNH.**")
            
        if slope_change > 0.15:
            st.warning("⚠️ **Cảnh báo Trôi (Drift):** Hệ số Slope biến động mạnh (>15%) trong khoảng thời gian này. Hệ thống thiếu ổn định.")
