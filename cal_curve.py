import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Roche Cal Expert", layout="wide", page_icon="🧪")

# --- 1. QUẢN LÝ SESSION STATE (Lưu dữ liệu tạm thời) ---
if 'master_params' not in st.session_state:
    # Mặc định load thông số Anti-TPO (Competitive)
    st.session_state.master_params = {'A': 876721.0, 'B': 0.762881, 'C': 175.289, 'D': -1315.11}

if 'cal_current_results' not in st.session_state:
    st.session_state.cal_current_results = None # Lưu kết quả Mode 1

# --- 2. CÁC HÀM TOÁN HỌC CỐT LÕI ---
def rod_4pl(x, A, B, C, D):
    """Tính Tín hiệu từ Nồng độ (Master Curve)"""
    if x < 0: return A # Xử lý nồng độ âm
    try:
        return D + (A - D) / (1.0 + (x / C) ** B)
    except:
        return np.nan

def inv_rod_4pl(y, A, B, C, D):
    """Tính Nồng độ từ Tín hiệu (Inverse 4PL)"""
    try:
        if (A - D) == 0 or (y - D) == 0: return np.nan
        term = (A - D) / (y - D) - 1
        if term <= 0: return np.nan
        return C * (term ** (1/B))
    except:
        return np.nan

# --- 3. SIDEBAR: CẤU HÌNH CHUNG ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # CHỌN CHẾ ĐỘ
    app_mode = st.radio(
        "Chọn Chức năng:",
        ["1. Tạo Cal & Tính mẫu", "2. Troubleshoot (Check Lịch sử)"],
        captions=["Chạy máy hàng ngày", "Phân tích xu hướng & Lỗi"]
    )
    
    st.divider()
    
    # NHẬP MASTER CURVE (Dùng chung cho cả 2 mode)
    st.subheader("Cấu hình Master Curve")
    st.caption("Thông số từ XML/Barcode hóa chất")
    
    # Input có lưu vào Session State
    mA = st.number_input("A (Signal tại 0)", value=st.session_state.master_params['A'], format="%.0f")
    mB = st.number_input("B (Slope)", value=st.session_state.master_params['B'], format="%.6f")
    mC = st.number_input("C (IC50)", value=st.session_state.master_params['C'], format="%.4f")
    mD = st.number_input("D (Signal vô cùng)", value=st.session_state.master_params['D'], format="%.0f")
    
    # Cập nhật state
    st.session_state.master_params = {'A': mA, 'B': mB, 'C': mC, 'D': mD}

# =========================================================
# MODE 1: TẠO ĐƯỜNG CAL VÀ TÍNH MẪU (CALCULATOR)
# =========================================================
if app_mode == "1. Tạo Cal & Tính mẫu":
    st.title("🧪 Mode 1: Calibration & Calculation")
    st.markdown("---")

    col_input, col_graph = st.columns([1, 1.5])

    with col_input:
        st.subheader("1. Nhập kết quả Calibrator")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Level 1 (Thấp)**")
            t1 = st.number_input("Target 1:", value=42.1)
            s1 = st.number_input("Signal 1:", value=583722.0)
        with c2:
            st.markdown("**Level 2 (Cao)**")
            t2 = st.number_input("Target 2:", value=372.0)
            s2 = st.number_input("Signal 2:", value=288320.0)

        if st.button("🚀 Dựng đường Cal (Recalibrate)", type="primary"):
            # Tính toán Logic
            p = st.session_state.master_params
            
            # 1. Tính Master Signal
            ms1 = rod_4pl(t1, **p)
            ms2 = rod_4pl(t2, **p)
            
            # 2. Tính Slope/Intercept
            if (ms2 - ms1) != 0:
                slope = (s2 - s1) / (ms2 - ms1)
                intercept = s1 - slope * ms1
                
                # Lưu kết quả
                st.session_state.cal_current_results = {
                    'slope': slope, 'intercept': intercept,
                    't1': t1, 't2': t2, 's1': s1, 's2': s2
                }
                st.success("Đã dựng đường chuẩn thành công!")
            else:
                st.error("Lỗi: Không thể tính toán (Mẫu số bằng 0). Kiểm tra lại Target.")

        # HIỂN THỊ KẾT QUẢ CAL & TÍNH MẪU
        if st.session_state.cal_current_results:
            res = st.session_state.cal_current_results
            st.divider()
            
            # Hiển thị thông số
            k1, k2, k3 = st.columns(3)
            k1.metric("Slope", f"{res['slope']:.4f}")
            k2.metric("Intercept", f"{res['intercept']:.0f}")
            status = "✅ PASS" if 0.8 <= res['slope'] <= 1.2 else "❌ FAIL"
            k3.metric("Đánh giá", status)

            # Form tính mẫu
            st.subheader("2. Tính mẫu bệnh nhân")
            with st.form("sample_calc"):
                sig_sample = st.number_input("Nhập Signal mẫu:", value=400000.0)
                btn_calc = st.form_submit_button("Tính kết quả")
                
                if btn_calc:
                    # Logic tính ngược
                    # B1: Chuẩn hóa signal
                    sig_norm = (sig_sample - res['intercept']) / res['slope']
                    # B2: Tra ngược Master
                    conc_result = inv_rod_4pl(sig_norm, **st.session_state.master_params)
                    
                    st.info(f"👉 Kết quả: **{conc_result:.4f}** (IU/mL)")

    with col_graph:
        if st.session_state.cal_current_results:
            res = st.session_state.cal_current_results
            p = st.session_state.master_params
            
            st.subheader("Biểu đồ Đường chuẩn")
            
            # Tạo dữ liệu vẽ
            x_plot = np.logspace(np.log10(5), np.log10(1000), 200)
            y_master = [rod_4pl(x, **p) for x in x_plot]
            y_actual = [y * res['slope'] + res['intercept'] for y in y_master]
            
            fig = go.Figure()
            # Master
            fig.add_trace(go.Scatter(x=x_plot, y=y_master, mode='lines', name='Master (Gốc)', line=dict(dash='dash', color='gray')))
            # Actual
            fig.add_trace(go.Scatter(x=x_plot, y=y_actual, mode='lines', name='Hiện tại', line=dict(color='blue')))
            # Points
            fig.add_trace(go.Scatter(
                x=[res['t1'], res['t2']], y=[res['s1'], res['s2']],
                mode='markers', name='Điểm Cal', marker=dict(size=12, color='red', symbol='cross')
            ))
            
            fig.update_layout(xaxis_type="log", yaxis_type="log", height=500, xaxis_title="Nồng độ", yaxis_title="Tín hiệu")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Vui lòng nhập dữ liệu và bấm nút Dựng đường Cal")

# =========================================================
# MODE 2: CHECK NHIỀU ĐƯỜNG CAL (TROUBLESHOOT)
# =========================================================
elif app_mode == "2. Troubleshoot (Check Lịch sử)":
    st.title("📈 Mode 2: Trend Analysis & Troubleshoot")
    st.markdown("Nhập lịch sử chạy Cal để kiểm tra độ ổn định của hệ thống.")
    
    # 1. BẢNG NHẬP LIỆU HÀNG LOẠT
    st.subheader("1. Dữ liệu Lịch sử (Data Entry)")
    
    # Dataframe mẫu
    df_template = pd.DataFrame([
        {"Date": "2023-12-01", "Target 1": 42.1, "Target 2": 372.0, "Signal 1": 590000, "Signal 2": 295000},
        {"Date": "2023-12-15", "Target 1": 42.1, "Target 2": 372.0, "Signal 1": 583602, "Signal 2": 289073},
        {"Date": "2023-12-25", "Target 1": 42.1, "Target 2": 372.0, "Signal 1": 550000, "Signal 2": 270000},
    ])
    
    edited_df = st.data_editor(df_template, num_rows="dynamic", use_container_width=True)
    
    if st.button("🔍 Phân tích dữ liệu", type="primary"):
        p = st.session_state.master_params
        results = []
        
        # XỬ LÝ DỮ LIỆU
        for idx, row in edited_df.iterrows():
            try:
                t1, t2 = float(row['Target 1']), float(row['Target 2'])
                s1, s2 = float(row['Signal 1']), float(row['Signal 2'])
                
                # Tính Master Signal
                m1 = rod_4pl(t1, **p)
                m2 = rod_4pl(t2, **p)
                
                # Tính Slope
                slope = (s2 - s1) / (m2 - m1)
                
                results.append({
                    "Date": row['Date'],
                    "Slope": slope,
                    "Signal 1": s1, "Signal 2": s2,
                    "Target 1": t1, "Target 2": t2
                })
            except:
                pass
        
        df_res = pd.DataFrame(results)
        
        # HIỂN THỊ BIỂU ĐỒ
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Biểu đồ Xu hướng Slope")
            st.caption("Theo dõi sự suy hao tín hiệu theo thời gian (Chuẩn: 1.0)")
            
            fig_trend = go.Figure()
            # Vùng chuẩn
            fig_trend.add_hrect(y0=0.8, y1=1.2, fillcolor="green", opacity=0.1, line_width=0)
            fig_trend.add_hline(y=1.0, line_dash="dash", line_color="green")
            
            # Đường xu hướng
            fig_trend.add_trace(go.Scatter(
                x=df_res['Date'], y=df_res['Slope'],
                mode='lines+markers', name='Slope Factor',
                line=dict(color='blue', width=3), marker=dict(size=10)
            ))
            fig_trend.update_layout(yaxis_title="Slope Factor (Measured/Master)", height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with c2:
            st.subheader("Biểu đồ Phân bố (Overlay)")
            st.caption("So sánh các điểm Cal thực tế với đường Master Curve gốc")
            
            # Vẽ Master Curve nền
            x_draw = np.logspace(np.log10(5), np.log10(1000), 200)
            y_master_draw = [rod_4pl(x, **p) for x in x_draw]
            
            fig_overlay = go.Figure()
            fig_overlay.add_trace(go.Scatter(x=x_draw, y=y_master_draw, mode='lines', name='Master Curve', line=dict(color='gray', width=1)))
            
            # Vẽ các lần chạy
            for i, row in df_res.iterrows():
                fig_overlay.add_trace(go.Scatter(
                    x=[row['Target 1'], row['Target 2']],
                    y=[row['Signal 1'], row['Signal 2']],
                    mode='lines+markers',
                    name=str(row['Date']),
                    opacity=0.5
                ))
            
            fig_overlay.update_layout(xaxis_type="log", yaxis_type="log", height=400)
            st.plotly_chart(fig_overlay, use_container_width=True)
            
        # KẾT LUẬN
        st.info("💡 **Gợi ý:** Nếu đường Slope (biểu đồ trái) đi xuống liên tục, kiểm tra lại thuốc thử hoặc bóng đèn quang kế.")
