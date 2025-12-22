import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Roche Immunoassay Calibrator", layout="wide")

st.title("🧬 Roche Immunoassay Calibration Simulator")
st.markdown("Mô phỏng dựng đường cong chuẩn (4-Parameter Logistic) và tính toán ngược nồng độ.")

# --- 1. ĐỊNH NGHĨA HÀM TOÁN HỌC ---
def logistic_4pl(x, A, B, C, D):
    return D + (A - D) / (1.0 + (x / C) ** B)

def inverse_logistic_4pl(y, A, B, C, D):
    try:
        if (A - D) == 0 or (y - D) == 0: return np.nan
        term = (A - D) / (y - D) - 1
        if term <= 0: return np.nan
        return C * (term ** (1/B))
    except:
        return np.nan

# --- 2. SIDEBAR & NHẬP LIỆU ---
with st.sidebar:
    st.header("1. Nhập Dữ liệu Cal")
    st.info("Nhập các điểm chuẩn (Calibrators) vào bảng dưới đây. Bạn có thể thêm/sửa hàng.")

    # Dữ liệu mẫu khởi tạo
    default_data = pd.DataFrame({
        "Result (Nồng độ)": [0.0, 0.5, 2.0, 10.0, 50.0, 100.0],
        "Signal (RLU/Abs)": [500, 1200, 4500, 25000, 110000, 200000]
    })

    # Widget nhập liệu dạng bảng
    df_input = st.data_editor(default_data, num_rows="dynamic", hide_index=True)

    # Nút action
    run_cal = st.button("🚀 Dựng Đường Cong", type="primary")

# --- XỬ LÝ CHÍNH ---
if run_cal or True: # Mặc định chạy lần đầu
    # Lấy dữ liệu từ bảng
    try:
        # Lọc bỏ các hàng trống hoặc không phải số
        df_clean = df_input.dropna().astype(float)
        x_data = df_clean["Result (Nồng độ)"].values
        y_data = df_clean["Signal (RLU/Abs)"].values

        # Sắp xếp lại theo nồng độ tăng dần để vẽ cho đẹp
        sorted_indices = np.argsort(x_data)
        x_data = x_data[sorted_indices]
        y_data = y_data[sorted_indices]

        # --- FITTING ---
        # Ước lượng tham số ban đầu (Heuristic)
        # Tránh log(0) bằng cách thay 0 bằng giá trị rất nhỏ epsilon
        x_data_log = x_data.copy()
        x_data_log[x_data_log == 0] = 1e-3 
        
        p0 = [min(y_data), 1.0, np.median(x_data_log), max(y_data)]
        
        # Chạy thuật toán tối ưu
        popt, pcov = curve_fit(logistic_4pl, x_data, y_data, p0, maxfev=10000)
        A, B, C, D = popt
        
        # Tính R^2 để đánh giá độ khớp
        residuals = y_data - logistic_4pl(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)

    except Exception as e:
        st.error(f"Không thể dựng đường cong. Vui lòng kiểm tra lại dữ liệu đầu vào.\nLỗi chi tiết: {e}")
        st.stop()

    # --- GIAO DIỆN CHÍNH (MAIN COLUMN) ---
    col_graph, col_calc = st.columns([2, 1])

    with col_graph:
        st.subheader("2. Biểu đồ Đường Chuẩn (Log-Log Scale)")
        
        # Tạo dữ liệu mượt cho đường cong
        x_min = max(1e-3, min(x_data[x_data > 0])) / 2
        x_max = max(x_data) * 2
        x_curve = np.logspace(np.log10(x_min), np.log10(x_max), 500)
        y_curve = logistic_4pl(x_curve, *popt)

        # Vẽ bằng Plotly
        fig = go.Figure()

        # 1. Điểm Cal thực tế
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            mode='markers',
            name='Cal Points (Thực tế)',
            marker=dict(size=12, color='red', line=dict(width=2, color='DarkSlateGrey'))
        ))

        # 2. Đường cong Fitted
        fig.add_trace(go.Scatter(
            x=x_curve, y=y_curve,
            mode='lines',
            name='Fitted Curve (4PL)',
            line=dict(color='blue', width=3)
        ))

        # Cấu hình trục Logarit (Đặc trưng miễn dịch)
        fig.update_layout(
            xaxis_type="log", yaxis_type="log",
            xaxis_title="Nồng độ (Result)",
            yaxis_title="Tín hiệu (Signal)",
            template="plotly_white",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Hiển thị tham số
        with st.expander("Xem chi tiết tham số phương trình"):
            st.latex(r"Signal = D + \frac{A - D}{1 + (\frac{Result}{C})^B}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("A (Min)", f"{A:.2f}")
            c2.metric("D (Max)", f"{D:.2f}")
            c3.metric("C (IC50)", f"{C:.2f}")
            c4.metric("B (Slope)", f"{B:.2f}")
            c5.metric("R² Fit", f"{r_squared:.4f}")

    with col_calc:
        st.subheader("3. Công cụ Tính toán")
        st.write("Nhập 1 thông số để tính thông số còn lại dựa trên đường cong bên cạnh.")

        calc_mode = st.radio("Chọn chiều tính:", ["Signal ➔ Result", "Result ➔ Signal"])
        
        result_val = None
        input_val = None

        if calc_mode == "Signal ➔ Result":
            input_val = st.number_input("Nhập Tín hiệu (Signal):", value=float(np.mean(y_data)))
            if st.button("Tính Nồng độ"):
                calc_res = inverse_logistic_4pl(input_val, *popt)
                if np.isnan(calc_res):
                    st.warning("⚠️ Tín hiệu nằm ngoài phạm vi đường cong (bão hòa hoặc thấp hơn nhiễu nền).")
                else:
                    st.success(f"📌 Nồng độ: **{calc_res:.4f}**")
                    result_val = calc_res # Lưu để vẽ điểm lên đồ thị
                    
                    # Cập nhật điểm vừa tính lên đồ thị
                    fig.add_trace(go.Scatter(
                        x=[calc_res], y=[input_val],
                        mode='markers', name='Kết quả vừa tính',
                        marker=dict(size=15, color='green', symbol='star')
                    ))
                    st.plotly_chart(fig, use_container_width=True) # Vẽ lại đồ thị với điểm mới

        else: # Result -> Signal
            input_val = st.number_input("Nhập Nồng độ (Result):", value=float(np.median(x_data)))
            if st.button("Tính Tín hiệu"):
                calc_sig = logistic_4pl(input_val, *popt)
                st.success(f"⚡ Tín hiệu: **{calc_sig:.2f}**")
                
                # Cập nhật điểm vừa tính lên đồ thị
                fig.add_trace(go.Scatter(
                    x=[input_val], y=[calc_sig],
                    mode='markers', name='Kết quả vừa tính',
                    marker=dict(size=15, color='orange', symbol='star')
                ))
                st.plotly_chart(fig, use_container_width=True)
