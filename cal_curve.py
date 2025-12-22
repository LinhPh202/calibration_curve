import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Universal Calibrator", layout="wide")
st.title("🧪 Universal Lab Calibration Tool")
st.markdown("Công cụ dựng đường chuẩn cho cả **Sinh hóa (Linear)** và **Miễn dịch (4PL/Rodbard)**.")

# --- 1. ĐỊNH NGHĨA CÁC HÀM TOÁN HỌC ---

# --- A. Mô hình 4PL (Miễn dịch / Sinh hóa Protein) ---
def func_4pl(x, A, B, C, D):
    return D + (A - D) / (1.0 + (x / C) ** B)

def inv_func_4pl(y, A, B, C, D):
    try:
        if (A - D) == 0 or (y - D) == 0: return np.nan
        term = (A - D) / (y - D) - 1
        if term <= 0: return np.nan
        return C * (term ** (1/B))
    except: return np.nan

# --- B. Mô hình Linear (Sinh hóa thường: Glu, Ure...) ---
def func_linear(x, slope, intercept):
    return slope * x + intercept

def inv_func_linear(y, slope, intercept):
    if slope == 0: return np.nan
    return (y - intercept) / slope

# --- 2. GIAO DIỆN & SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    # CHỌN MÔ HÌNH
    cal_model = st.selectbox(
        "Chọn Mô hình Cal:",
        ("Linear (Tuyến tính)", "Rodbard (4PL)")
    )
    
    st.info("""
    * **Linear:** Glucose, Ure, Cre, AST, ALT...
    * **Rodbard (4PL):** TSH, Ferritin, Troponin, CRP, HbA1c...
    """)

    st.divider()
    st.header("📝 Dữ liệu Cal")

    # Dữ liệu mẫu thay đổi theo mô hình
    if cal_model == "Linear (Tuyến tính)":
        default_data = pd.DataFrame({
            "Result (Nồng độ)": [0.0, 100.0], # Thường chỉ cần 2 điểm (Blank + Standard)
            "Signal (Abs/OD)": [0.005, 1.250]
        })
    else:
        default_data = pd.DataFrame({
            "Result (Nồng độ)": [0.0, 0.5, 5.0, 50.0, 100.0],
            "Signal (RLU)": [400, 1000, 8000, 120000, 210000]
        })

    df_input = st.data_editor(default_data, num_rows="dynamic", hide_index=True)
    
    run_cal = st.button("🚀 Dựng Đường Cong", type="primary")

# --- XỬ LÝ CHÍNH ---
if run_cal or True:
    try:
        df_clean = df_input.dropna().astype(float)
        x_data = df_clean["Result (Nồng độ)"].values
        y_data = df_clean["Signal (RLU)" if "RLU" in df_clean.columns else "Signal (Abs/OD)"].values
        
        # Sắp xếp dữ liệu
        idx = np.argsort(x_data)
        x_data = x_data[idx]
        y_data = y_data[idx]

        popt = None
        r_squared = 0
        model_name = ""

        # --- FITTING LOGIC ---
        if cal_model == "Linear (Tuyến tính)":
            model_name = "Linear Regression (Y = Ax + B)"
            # Dùng numpy polyfit cho phương trình bậc 1
            slope, intercept = np.polyfit(x_data, y_data, 1)
            popt = (slope, intercept)
            
            # Tính R^2
            residuals = y_data - func_linear(x_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        else: # 4PL
            model_name = "Rodbard 4-Parameter Logistic"
            x_log = x_data.copy()
            x_log[x_log == 0] = 1e-3
            p0 = [min(y_data), 1.0, np.median(x_log), max(y_data)]
            popt, _ = curve_fit(func_4pl, x_data, y_data, p0, maxfev=10000)
            
            # Tính R^2
            residuals = y_data - func_4pl(x_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot)

    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
        st.stop()

    # --- HIỂN THỊ KẾT QUẢ ---
    col_graph, col_calc = st.columns([2, 1])

    with col_graph:
        st.subheader(f"Biểu đồ: {model_name}")
        
        # Tạo điểm vẽ đường cong mịn
        if cal_model == "Linear (Tuyến tính)":
            x_curve = np.linspace(0, max(x_data)*1.2, 100)
            y_curve = func_linear(x_curve, *popt)
            log_scale = False
        else:
            x_min = max(1e-3, min(x_data[x_data > 0])) / 2
            x_max = max(x_data) * 1.5
            x_curve = np.logspace(np.log10(x_min), np.log10(x_max), 500)
            y_curve = func_4pl(x_curve, *popt)
            log_scale = True

        # Vẽ Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Điểm Cal', marker=dict(color='red', size=10)))
        fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='Đường chuẩn', line=dict(color='blue')))

        layout_args = dict(
            xaxis_title="Nồng độ (Result)", yaxis_title="Tín hiệu (Signal)",
            template="plotly_white", height=500
        )
        # Chỉ dùng log scale cho 4PL, Linear để thường dễ nhìn hơn
        if log_scale:
            layout_args.update(xaxis_type="log", yaxis_type="log")
            
        fig.update_layout(**layout_args)
        st.plotly_chart(fig, use_container_width=True)

        # Hiển thị tham số
        with st.expander("Tham số phương trình"):
            if cal_model == "Linear (Tuyến tính)":
                st.latex(r"Signal = Slope \times Result + Intercept")
                st.write(f"**Slope (Hệ số góc):** {popt[0]:.4f}")
                st.write(f"**Intercept (Hệ số chặn):** {popt[1]:.4f}")
            else:
                st.write(f"A={popt[0]:.2f}, B={popt[1]:.2f}, C={popt[2]:.2f}, D={popt[3]:.2f}")
            st.metric("Độ khớp (R²)", f"{r_squared:.4f}")

    with col_calc:
        st.subheader("Tính toán")
        calc_mode = st.radio("Chiều tính:", ["Signal ➔ Result", "Result ➔ Signal"])
        
        val = st.number_input("Nhập giá trị:", value=0.0, format="%.4f")
        
        if st.button("Tính ngay"):
            res = None
            if cal_model == "Linear (Tuyến tính)":
                if calc_mode == "Signal ➔ Result":
                    res = inv_func_linear(val, *popt)
                    st.success(f"Nồng độ: {res:.4f}")
                    fig.add_trace(go.Scatter(x=[res], y=[val], mode='markers', marker=dict(color='green', size=15, symbol='star'), name='Điểm tính'))
                else:
                    res = func_linear(val, *popt)
                    st.success(f"Tín hiệu: {res:.4f}")
                    fig.add_trace(go.Scatter(x=[val], y=[res], mode='markers', marker=dict(color='orange', size=15, symbol='star'), name='Điểm tính'))
            
            else: # 4PL
                if calc_mode == "Signal ➔ Result":
                    res = inv_func_4pl(val, *popt)
                    if np.isnan(res): st.warning("Ngoài phạm vi đo")
                    else: 
                        st.success(f"Nồng độ: {res:.4f}")
                        fig.add_trace(go.Scatter(x=[res], y=[val], mode='markers', marker=dict(color='green', size=15, symbol='star'), name='Điểm tính'))
                else:
                    res = func_4pl(val, *popt)
                    st.success(f"Tín hiệu: {res:.4f}")
                    fig.add_trace(go.Scatter(x=[val], y=[res], mode='markers', marker=dict(color='orange', size=15, symbol='star'), name='Điểm tính'))
            
            st.plotly_chart(fig, use_container_width=True)
