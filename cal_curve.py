import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Roche Recalibration Tool", layout="wide")
st.title("🎛️ Roche Master Curve Recalibration")
st.markdown("""
Quy trình:
1. Nhập tham số **Master Curve** (A, B, C, D) từ nhà sản xuất.
2. Nhập kết quả chạy **Cal 2 điểm** thực tế tại phòng Lab.
3. Hệ thống sẽ **Recalibrate** (nắn đường cong) và tính kết quả mẫu.
""")

# --- 1. HÀM TOÁN HỌC (RODBARD 4PL) ---
def rodbard_4pl(x, A, B, C, D):
    """Tính Tín hiệu (Signal) từ Nồng độ (x) dựa trên Master Curve"""
    # Công thức: Signal = D + (A - D) / (1 + (x/C)^B)
    # Lưu ý: Với Roche, đôi khi A là Max, D là Min hoặc ngược lại. 
    # Hàm này viết theo dạng tổng quát.
    try:
        return D + (A - D) / (1.0 + (x / C) ** B)
    except:
        return np.nan

def inv_rodbard_4pl(y, A, B, C, D):
    """Tính Nồng độ (x) từ Tín hiệu (y)"""
    try:
        if (A - D) == 0 or (y - D) == 0: return np.nan
        term = (A - D) / (y - D) - 1
        if term <= 0: return np.nan # Lỗi toán học (căn bậc chẵn của số âm)
        return C * (term ** (1/B))
    except:
        return np.nan

# --- 2. GIAO DIỆN NHẬP LIỆU ---

# Cột trái: Nhập tham số Master Curve
with st.sidebar:
    st.header("1. Master Curve Parameters")
    st.info("Nhập tham số từ file XML hoặc Barcode tờ hóa chất.")
    
    # Giá trị mặc định lấy từ ví dụ XML Anti-TPO bạn cung cấp
    # XML: "876721 175.289 0.762881 -1315.11"
    # Mapping phỏng đoán: A=Min, B=Slope, C=IC50, D=Max (hoặc đảo A/D)
    
    param_A = st.number_input("Tham số A (Signal tại Conc 0/Min)", value=-1315.0, format="%.2f")
    param_B = st.number_input("Tham số B (Hệ số dốc - Slope)", value=0.762881, format="%.6f")
    param_C = st.number_input("Tham số C (Điểm uốn - IC50)", value=175.289, format="%.4f")
    param_D = st.number_input("Tham số D (Signal tại Max/Inf)", value=876721.0, format="%.2f")
    
    st.markdown("---")
    st.caption("Gợi ý từ XML Anti-TPO của bạn:\nA=-1315, B=0.76, C=175, D=876721")

# Khu vực chính: Nhập kết quả Cal thực tế
st.header("2. Nhập kết quả Calibrator tại Lab")
col_cal1, col_cal2 = st.columns(2)

with col_cal1:
    st.subheader("Calibrator 1 (Thấp)")
    cal1_target = st.number_input("Nồng độ Target (Cal 1):", value=0.0, min_value=0.0)
    # Cal 1 thực tế có thể khác Master (Master nền âm, thực tế nền dương khoảng 500-1000)
    cal1_actual_sig = st.number_input("Tín hiệu đo được (Signal 1):", value=1500.0) 

with col_cal2:
    st.subheader("Calibrator 2 (Cao)")
    cal2_target = st.number_input("Nồng độ Target (Cal 2):", value=175.0) # Thường target gần điểm uốn
    # Tín hiệu đo được thực tế (Ví dụ thuốc thử yếu đi chút so với Master)
    cal2_actual_sig = st.number_input("Tín hiệu đo được (Signal 2):", value=400000.0)

# --- 3. XỬ LÝ RECALIBRATION ---
st.divider()

# Bước A: Tính tín hiệu LÝ THUYẾT trên Master Curve tại 2 nồng độ Target
# Xử lý trường hợp nồng độ 0 cho hàm log (thay bằng số rất nhỏ)
c1_calc = cal1_target if cal1_target > 1e-5 else 1e-5
c2_calc = cal2_target if cal2_target > 1e-5 else 1e-5

master_sig_1 = rodbard_4pl(c1_calc, param_A, param_B, param_C, param_D)
master_sig_2 = rodbard_4pl(c2_calc, param_A, param_B, param_C, param_D)

# Bước B: Tìm phương trình biến đổi tuyến tính (Linear Mapping)
# Actual_Signal = Slope * Master_Signal + Intercept
if (master_sig_2 - master_sig_1) == 0:
    st.error("Lỗi: Hai điểm Cal có tín hiệu Master giống hệt nhau. Vui lòng kiểm tra nồng độ.")
    st.stop()

slope = (cal2_actual_sig - cal1_actual_sig) / (master_sig_2 - master_sig_1)
intercept = cal1_actual_sig - slope * master_sig_1

# Hiển thị thông tin Cal
col_res1, col_res2 = st.columns([1, 2])
with col_res1:
    st.subheader("Kết quả Recalibration")
    st.metric("Hệ số góc (Slope)", f"{slope:.4f}", help="Tỷ lệ tín hiệu Thực tế / Master. Tốt nhất trong khoảng 0.8 - 1.2")
    st.metric("Điểm chặn (Intercept)", f"{intercept:.2f}", help="Độ lệch nền tín hiệu.")
    
    status = "✅ ĐẠT (Passed)" if 0.8 <= slope <= 1.2 else "⚠️ CẢNH BÁO (Check)"
    st.write(f"Trạng thái: **{status}**")

# --- 4. VẼ BIỂU ĐỒ ---
with col_res2:
    # Tạo dữ liệu vẽ
    x_draw = np.logspace(np.log10(0.1), np.log10(1000), 200)
    
    # 1. Đường Master Curve (Gốc)
    y_master = rodbard_4pl(x_draw, param_A, param_B, param_C, param_D)
    
    # 2. Đường Recalibrated (Đường dùng cho mẫu bệnh nhân)
    # Tín hiệu tại mỗi điểm nồng độ x sẽ bị biến đổi theo slope & intercept
    y_recal = y_master * slope + intercept
    
    fig = go.Figure()
    
    # Vẽ Master
    fig.add_trace(go.Scatter(x=x_draw, y=y_master, mode='lines', name='Master Curve (Gốc)', line=dict(dash='dash', color='gray')))
    
    # Vẽ Recalibrated
    fig.add_trace(go.Scatter(x=x_draw, y=y_recal, mode='lines', name='Recalibrated (Thực tế)', line=dict(color='blue', width=3)))
    
    # Vẽ 2 điểm Cal thực tế
    fig.add_trace(go.Scatter(
        x=[cal1_target if cal1_target>0 else 0.1, cal2_target], 
        y=[cal1_actual_sig, cal2_actual_sig],
        mode='markers', name='Điểm Cal Lab', marker=dict(color='red', size=12, symbol='x')
    ))

    fig.update_layout(
        title="So sánh Đường chuẩn Gốc và Thực tế",
        xaxis_title="Nồng độ (Log scale)",
        yaxis_title="Tín hiệu (Signal)",
        xaxis_type="log", yaxis_type="log",
        height=450, margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 5. TÍNH KẾT QUẢ MẪU (BỆNH NHÂN) ---
st.divider()
st.header("3. Tính kết quả mẫu (Sample Calculator)")

col_input, col_output = st.columns(2)
with col_input:
    sample_signal = st.number_input("Nhập Tín hiệu mẫu (RLU/Counts):", value=50000.0)
    
    st.markdown("""
    **Công thức chuyển đổi:**
    1. **Chuẩn hóa:** $Sig_{Master} = (Sig_{Lab} - Intercept) / Slope$
    2. **Tra ngược:** $Result = f^{-1}(Sig_{Master}, A, B, C, D)$
    """)

with col_output:
    if st.button("Tính kết quả ngay"):
        # B1: Chuyển đổi Signal Lab -> Signal Master tương đương
        if slope == 0:
            st.error("Lỗi: Slope = 0")
        else:
            sig_normalized = (sample_signal - intercept) / slope
            
            # B2: Tính nồng độ từ Signal Master bằng tham số A,B,C,D gốc
            final_result = inv_rodbard_4pl(sig_normalized, param_A, param_B, param_C, param_D)
            
            if np.isnan(final_result):
                st.warning("⚠️ Không tính được kết quả (Tín hiệu ngoài dải đo hoặc lỗi toán học).")
            else:
                st.success(f"KẾT QUẢ: **{final_result:.4f}**")
                st.caption(f"(Tín hiệu quy đổi về Master: {sig_normalized:.2f})")
                
                # Vẽ điểm mẫu lên đồ thị
                fig.add_trace(go.Scatter(
                    x=[final_result], y=[sample_signal],
                    mode='markers', name='Mẫu vừa tính',
                    marker=dict(color='green', size=15, symbol='star')
                ))
                st.plotly_chart(fig, use_container_width=True)
