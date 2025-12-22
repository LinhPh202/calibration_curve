import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Roche Advanced Calibrator", layout="wide")
st.title("🧬 Roche Advanced Calibration Tool")
st.markdown("""
Công cụ hỗ trợ đầy đủ các dạng:
* **Sinh hóa Tuyến tính:** Glucose, Ure (Input: 2 Abs, Model: Linear)
* **Sinh hóa Miễn dịch/Độ đục:** CRP, HbA1c (Input: 2 Abs, Model: 4PL)
* **Miễn dịch ECLIA:** TSH, FT4 (Input: 1 Signal, Model: 4PL)
""")

# --- 1. ĐỊNH NGHĨA HÀM TOÁN HỌC ---
def func_4pl(x, A, B, C, D):
    return D + (A - D) / (1.0 + (x / C) ** B)

def inv_func_4pl(y, A, B, C, D):
    try:
        if (A - D) == 0 or (y - D) == 0: return np.nan
        term = (A - D) / (y - D) - 1
        if term <= 0: return np.nan
        return C * (term ** (1/B))
    except: return np.nan

def func_linear(x, slope, intercept):
    return slope * x + intercept

def inv_func_linear(y, slope, intercept):
    if slope == 0: return np.nan
    return (y - intercept) / slope

# --- 2. SIDEBAR CẤU HÌNH ---
with st.sidebar:
    st.header("1. Cấu hình Input (Đầu vào)")
    
    # BƯỚC 1: CHỌN CÁCH NHẬP LIỆU (SINH HÓA vs MIỄN DỊCH)
    input_mode = st.radio(
        "Nguồn dữ liệu:",
        ("Sinh hóa (2 điểm Abs)", "Miễn dịch (1 điểm Signal)")
    )
    
    calc_method = "None"
    if input_mode == "Sinh hóa (2 điểm Abs)":
        st.caption("Nhập Raw Absorbance từ máy (Main + Sub/Blank)")
        calc_method = st.selectbox(
            "Cách tính Delta Abs:",
            ("Abs 2 - Abs 1 (Tăng quang)", "Abs 1 - Abs 2 (Giảm quang)")
        )

    st.divider()
    
    st.header("2. Cấu hình Model (Toán học)")
    # BƯỚC 2: CHỌN MÔ HÌNH TOÁN HỌC
    # Miễn dịch mặc định là 4PL, nhưng Sinh hóa có thể chọn Linear hoặc 4PL
    model_options = ["Linear (Tuyến tính)", "Rodbard (4PL / Non-Linear)"]
    if input_mode == "Miễn dịch (1 điểm Signal)":
        cal_model = "Rodbard (4PL / Non-Linear)" # Miễn dịch luôn cong
        st.info("Miễn dịch mặc định dùng mô hình Rodbard 4PL.")
    else:
        cal_model = st.selectbox("Chọn mô hình đường chuẩn:", model_options)
        if cal_model == "Linear (Tuyến tính)":
            st.caption("Dùng cho: Glu, Ure, Cre, AST, ALT...")
        else:
            st.caption("Dùng cho: CRP, HbA1c, RF, ASO, IgM...")

    st.divider()

    # BƯỚC 3: DATA EDITOR
    st.header("3. Dữ liệu Cal")
    
    if input_mode == "Sinh hóa (2 điểm Abs)":
        # Data mẫu cho Sinh hóa
        if cal_model == "Linear (Tuyến tính)":
             # Mẫu Linear (ít điểm)
            default_data = pd.DataFrame({
                "Result": [0.0, 100.0],
                "Abs 1":  [0.05, 0.05],
                "Abs 2":  [0.06, 0.80]
            })
        else:
            # Mẫu Non-Linear (CRP - Nhiều điểm)
            default_data = pd.DataFrame({
                "Result": [0.0, 5.0, 20.0, 80.0, 160.0, 320.0],
                "Abs 1":  [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                "Abs 2":  [0.03, 0.10, 0.40, 1.20, 1.80, 2.10] # Bão hòa dần
            })
    else:
        # Mẫu Miễn dịch
        default_data = pd.DataFrame({
            "Result": [0.0, 0.5, 5.0, 50.0, 100.0],
            "Signal": [400, 1200, 8500, 120000, 210000]
        })

    df_input = st.data_editor(default_data, num_rows="dynamic", hide_index=True)
    run_cal = st.button("🚀 Dựng Đường Cong", type="primary")

# --- 3. XỬ LÝ LOGIC ---
if run_cal or True:
    try:
        df_clean = df_input.dropna().astype(float)
        x_data = df_clean["Result"].values
        
        # Xử lý Y-Data (Delta Abs hoặc Signal)
        y_label = ""
        if input_mode == "Sinh hóa (2 điểm Abs)":
            abs1 = df_clean["Abs 1"].values
            abs2 = df_clean["Abs 2"].values
            if "Abs 2 - Abs 1" in calc_method:
                y_data = abs2 - abs1
            else:
                y_data = abs1 - abs2
            y_label = "Delta Absorbance"
        else:
            y_data = df_clean["Signal"].values
            y_label = "Signal (RLU/Counts)"

        # Sort
        idx = np.argsort(x_data)
        x_data = x_data[idx]
        y_data = y_data[idx]

        # Fitting Variables
        popt = None
        r_squared = 0
        
        # --- THUẬT TOÁN FITTING ---
        if cal_model == "Linear (Tuyến tính)":
            slope, intercept = np.polyfit(x_data, y_data, 1)
            popt = (slope, intercept)
            
            # Tính R2
            residuals = y_data - func_linear(x_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        else: # Rodbard 4PL
            # Ước lượng tham số ban đầu (Quan trọng cho Sinh hóa vì số nhỏ)
            # Với Sinh hóa, Abs max chỉ tầm 2.0-3.0, không phải hàng nghìn như miễn dịch
            x_log = x_data.copy()
            x_log[x_log == 0] = 1e-4 # Tránh log(0)
            
            p0 = [min(y_data), 1.0, np.median(x_log), max(y_data)]
            
            # Chạy fitting
            popt, pcov = curve_fit(func_4pl, x_data, y_data, p0, maxfev=20000)
            
            residuals = y_data - func_4pl(x_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot)

    except Exception as e:
        st.error(f"Không thể dựng đường cong. Lỗi: {e}")
        st.stop()

    # --- 4. HIỂN THỊ BIỂU ĐỒ ---
    col_graph, col_calc = st.columns([2, 1])

    with col_graph:
        st.subheader("Biểu đồ Đường chuẩn")
        
        fig = go.Figure()

        # Vẽ điểm gốc
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, mode='markers', name='Cal Points',
            marker=dict(color='red', size=12, line=dict(width=1, color='black'))
        ))

        # Vẽ đường Fit
        if cal_model == "Linear (Tuyến tính)":
            x_curve = np.linspace(0, max(x_data)*1.1, 100)
            y_curve = func_linear(x_curve, *popt)
            fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='Linear Fit', line=dict(color='blue')))
            
            # Linear dùng trục thường
            fig.update_layout(xaxis_type="linear", yaxis_type="linear")
        
        else: # 4PL
            # Tạo dải X mượt (logspace)
            x_min_plot = max(1e-3, min(x_data[x_data>0])) / 2
            x_max_plot = max(x_data) * 1.5
            x_curve = np.logspace(np.log10(x_min_plot), np.log10(x_max_plot), 500)
            y_curve = func_4pl(x_curve, *popt)
            
            fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='4PL Fit', line=dict(color='blue')))
            
            # 4PL thường dùng trục Log-Log hoặc Linear-Linear tùy người xem
            # Ở đây để Log cho X, Linear cho Y (Semi-log) thường dùng trong sinh hóa miễn dịch
            # Hoặc Log-Log nếu dải đo rộng. Tôi sẽ để Log-Log mặc định.
            fig.update_layout(xaxis_type="log", yaxis_type="log" if input_mode!="Sinh hóa (2 điểm Abs)" else "linear") 

        fig.update_layout(
            title=f"Model: {cal_model} | R²: {r_squared:.4f}",
            xaxis_title="Nồng độ (Concentration)",
            yaxis_title=y_label,
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 5. CÔNG CỤ TÍNH TOÁN ---
    with col_calc:
        st.subheader("Tính mẫu (Interpolation)")
        st.caption(f"Đang dùng mô hình: **{cal_model}**")
        
        # INPUT CHO TÍNH TOÁN
        input_val_calc = 0.0
        
        if input_mode == "Sinh hóa (2 điểm Abs)":
            c1, c2 = st.columns(2)
            p_abs1 = c1.number_input("Abs 1 (Sample)", value=0.0, format="%.4f")
            p_abs2 = c2.number_input("Abs 2 (Sample)", value=0.0, format="%.4f")
            
            if "Abs 2 - Abs 1" in calc_method:
                input_val_calc = p_abs2 - p_abs1
            else:
                input_val_calc = p_abs1 - p_abs2
            
            st.info(f"Delta Abs tính được: **{input_val_calc:.4f}**")
        else:
            input_val_calc = st.number_input("Nhập Signal (Sample)", value=0.0)

        # NÚT TÍNH
        if st.button("Tính kết quả"):
            res = np.nan
            if cal_model == "Linear (Tuyến tính)":
                res = inv_func_linear(input_val_calc, *popt)
            else: # 4PL
                res = inv_func_4pl(input_val_calc, *popt)
            
            if np.isnan(res) or res < 0:
                st.warning("⚠️ Không tính được (Ngoài phạm vi hoặc tín hiệu âm).")
            else:
                st.success(f"Nồng độ: **{res:.4f}**")
                
                # Vẽ điểm mẫu lên đồ thị
                fig.add_trace(go.Scatter(
                    x=[res], y=[input_val_calc],
                    mode='markers', name='Kết quả mẫu',
                    marker=dict(color='green', size=15, symbol='star')
                ))
                st.plotly_chart(fig, use_container_width=True)
