import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Roche Lab Calibrator (2-Abs)", layout="wide")
st.title("🧪 Lab Calibration Tool (Hỗ trợ 2 điểm Abs)")
st.markdown("Công cụ dựng đường chuẩn chuyên dụng cho **Sinh hóa (2-Abs Check)** và **Miễn dịch (4PL)**.")

# --- 1. ĐỊNH NGHĨA CÁC HÀM TOÁN HỌC ---

# --- A. Mô hình 4PL (Miễn dịch / Turbidimetry) ---
def func_4pl(x, A, B, C, D):
    # Tránh lỗi chia cho 0 hoặc mũ số âm trong tính toán phức tạp
    return D + (A - D) / (1.0 + (x / C) ** B)

def inv_func_4pl(y, A, B, C, D):
    try:
        if (A - D) == 0 or (y - D) == 0: return np.nan
        term = (A - D) / (y - D) - 1
        if term <= 0: return np.nan
        return C * (term ** (1/B))
    except: return np.nan

# --- B. Mô hình Linear (Sinh hóa thường) ---
def func_linear(x, slope, intercept):
    return slope * x + intercept

def inv_func_linear(y, slope, intercept):
    if slope == 0: return np.nan
    return (y - intercept) / slope

# --- 2. GIAO DIỆN & SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình Xét nghiệm")
    
    # CHỌN LOẠI XÉT NGHIỆM
    test_type = st.selectbox(
        "Loại xét nghiệm:",
        ("Sinh hóa (Photometric)", "Miễn dịch (ECLIA)")
    )
    
    # Cấu hình chi tiết cho Sinh hóa
    calc_method = "Standard"
    if test_type == "Sinh hóa (Photometric)":
        st.subheader("Công thức tính Abs")
        calc_method = st.radio(
            "Cách tính Delta Abs:",
            ("Abs 2 - Abs 1 (Tăng quang)", "Abs 1 - Abs 2 (Giảm quang)")
        )
        st.caption("Ví dụ: Glucose, Ure thường tăng quang. AST, ALT thường giảm quang (đo NADH giảm).")

    st.divider()
    st.header("📝 Nhập dữ liệu Cal")

    # TẠO BẢNG NHẬP LIỆU DỰA TRÊN LOẠI XÉT NGHIỆM
    if test_type == "Sinh hóa (Photometric)":
        # Sinh hóa: Cần nhập 2 điểm Abs
        default_data = pd.DataFrame({
            "Result (Nồng độ)": [0.0, 50.0, 100.0, 200.0, 400.0],
            "First Abs (A1)":   [0.010, 0.012, 0.015, 0.020, 0.025], # Điểm đo sớm (hoặc đo bước sóng phụ)
            "Second Abs (A2)":  [0.015, 0.250, 0.500, 1.000, 2.000]  # Điểm đo muộn (hoặc bước sóng chính)
        })
        st.info("Nhập **First Abs** và **Second Abs** từ máy (Raw Data). Hệ thống sẽ tự tính Delta Abs.")
    else:
        # Miễn dịch: Nhập 1 Signal (RLU)
        default_data = pd.DataFrame({
            "Result (Nồng độ)": [0.0, 0.5, 5.0, 50.0, 100.0],
            "Signal (RLU)": [400, 1000, 8000, 120000, 210000]
        })
        st.info("Nhập tín hiệu RLU/Counts cuối cùng.")

    df_input = st.data_editor(default_data, num_rows="dynamic", hide_index=True)
    
    run_cal = st.button("🚀 Dựng Đường Cong", type="primary")

# --- 3. XỬ LÝ DỮ LIỆU & TÍNH TOÁN ---
if run_cal or True:
    try:
        df_clean = df_input.dropna().astype(float)
        
        # LẤY DỮ LIỆU X VÀ Y
        x_data = df_clean["Result (Nồng độ)"].values
        
        if test_type == "Sinh hóa (Photometric)":
            # Xử lý 2 cột Abs
            abs1 = df_clean["First Abs (A1)"].values
            abs2 = df_clean["Second Abs (A2)"].values
            
            # Tính Delta Abs (Tín hiệu thực dùng để vẽ)
            if calc_method == "Abs 2 - Abs 1 (Tăng quang)":
                y_data = abs2 - abs1
            else:
                y_data = abs1 - abs2
            
            # Mô hình mặc định cho Sinh hóa là Tuyến tính (Linear) 
            # (Lưu ý: Một số xét nghiệm sinh hóa đặc biệt như CRP vẫn dùng 4PL, 
            # nhưng ở đây ta mặc định Linear cho phổ biến, hoặc có thể thêm tùy chọn chọn mô hình)
            model_type = "Linear" 
            
        else:
            # Miễn dịch
            y_data = df_clean["Signal (RLU)"].values
            model_type = "4PL"

        # Sắp xếp lại dữ liệu
        idx = np.argsort(x_data)
        x_data = x_data[idx]
        y_data = y_data[idx]

        # --- FITTING ---
        popt = None
        r_squared = 0
        
        if model_type == "Linear":
            # Hồi quy tuyến tính: y = ax + b
            slope, intercept = np.polyfit(x_data, y_data, 1)
            popt = (slope, intercept)
            
            # R^2
            residuals = y_data - func_linear(x_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
        else: # 4PL
            # Hồi quy Rodbard
            x_log = x_data.copy()
            x_log[x_log == 0] = 1e-3 # Tránh log(0)
            p0 = [min(y_data), 1.0, np.median(x_log), max(y_data)]
            popt, _ = curve_fit(func_4pl, x_data, y_data, p0, maxfev=10000)
            
            residuals = y_data - func_4pl(x_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot)

    except Exception as e:
        st.error(f"Lỗi dữ liệu: {e}")
        st.stop()

    # --- 4. HIỂN THỊ KẾT QUẢ ---
    col_graph, col_calc = st.columns([2, 1])

    with col_graph:
        st.subheader("Biểu đồ Đường Chuẩn")
        
        # Vẽ đường cong mịn
        if model_type == "Linear":
            x_curve = np.linspace(0, max(x_data)*1.1, 100)
            y_curve = func_linear(x_curve, *popt)
            x_title = "Nồng độ"
            y_title = "Delta Abs (Hiệu số mật độ quang)"
        else:
            x_min = max(1e-3, min(x_data[x_data > 0])) / 2
            x_max = max(x_data) * 1.5
            x_curve = np.logspace(np.log10(x_min), np.log10(x_max), 500)
            y_curve = func_4pl(x_curve, *popt)
            x_title = "Nồng độ (Log scale)"
            y_title = "Signal (Log scale)"

        fig = go.Figure()
        
        # Điểm dữ liệu gốc
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, 
            mode='markers', name='Điểm Cal (Tính toán)',
            marker=dict(color='red', size=12, line=dict(width=1, color='black')),
            hovertemplate="Conc: %{x}<br>Delta Signal: %{y:.4f}"
        ))
        
        # Đường cong Fit
        fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='Fitted Curve', line=dict(color='blue')))

        layout_args = dict(xaxis_title=x_title, yaxis_title=y_title, template="plotly_white", height=500)
        if model_type == "4PL":
            layout_args.update(xaxis_type="log", yaxis_type="log")
            
        fig.update_layout(**layout_args)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị tham số
        with st.expander("Chi tiết phương trình", expanded=True):
            if model_type == "Linear":
                st.latex(r"\Delta Abs = Slope \times Conc + Intercept")
                st.write(f"**Slope:** {popt[0]:.5f} | **Intercept:** {popt[1]:.5f} | **R²:** {r_squared:.4f}")
            else:
                st.write(f"A={popt[0]:.2f}, B={popt[1]:.2f}, C={popt[2]:.2f}, D={popt[3]:.2f}")
                st.write(f"**R²:** {r_squared:.4f}")

    # --- 5. CÔNG CỤ TÍNH TOÁN ---
    with col_calc:
        st.subheader("Tính mẫu bệnh nhân")
        
        if test_type == "Sinh hóa (Photometric)":
            st.markdown("Nhập 2 giá trị Abs của mẫu bệnh nhân:")
            p_abs1 = st.number_input("Abs 1 (Bệnh nhân):", format="%.4f")
            p_abs2 = st.number_input("Abs 2 (Bệnh nhân):", format="%.4f", value=0.1)
            
            # Tự động tính Delta cho bệnh nhân theo quy tắc đã chọn
            if calc_method == "Abs 2 - Abs 1 (Tăng quang)":
                val_calc = p_abs2 - p_abs1
            else:
                val_calc = p_abs1 - p_abs2
                
            st.info(f"👉 Delta Abs tính toán: **{val_calc:.4f}**")
            
            if st.button("Tính Nồng độ"):
                res = inv_func_linear(val_calc, *popt)
                st.success(f"**Kết quả: {res:.4f}**")
                # Vẽ điểm lên đồ thị
                fig.add_trace(go.Scatter(x=[res], y=[val_calc], mode='markers', marker=dict(color='green', size=15, symbol='star'), name='Mẫu BN'))
                st.plotly_chart(fig, use_container_width=True)

        else: # Miễn dịch
            val_calc = st.number_input("Nhập Signal (RLU):", value=1000.0)
            if st.button("Tính Nồng độ"):
                res = inv_func_4pl(val_calc, *popt)
                if np.isnan(res): st.warning("Ngoài dải đo")
                else:
                    st.success(f"**Kết quả: {res:.4f}**")
                    fig.add_trace(go.Scatter(x=[res], y=[val_calc], mode='markers', marker=dict(color='green', size=15, symbol='star'), name='Mẫu BN'))
                    st.plotly_chart(fig, use_container_width=True)
