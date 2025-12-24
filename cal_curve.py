import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xml.etree.ElementTree as ET

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Roche Cal Expert Ultra", layout="wide", page_icon="🧪")

# ==============================================================================
# 1. QUẢN LÝ SESSION STATE
# ==============================================================================
if 'master_params' not in st.session_state:
    st.session_state.master_params = {'A': 876721.0, 'B': 0.762881, 'C': 175.289, 'D': -1315.11}

if 'qual_params' not in st.session_state:
    st.session_state.qual_params = {
        'FNeg': 1.0, 'FPos': 0.65, 'Const': 0.0,
        'MinNeg': 400.0, 'MaxNeg': 3500.0,
        'MinPos': 18000.0, 'MaxPos': 130000.0,
        'MinDiff': 16000.0
    }

if 'quant_results' not in st.session_state: st.session_state.quant_results = None
if 'qual_results' not in st.session_state: st.session_state.qual_results = None

# ==============================================================================
# 2. HÀM TOÁN HỌC & XỬ LÝ XML
# ==============================================================================
def rod_4pl(x, A, B, C, D):
    """Tính Tín hiệu từ Nồng độ"""
    if x < 0: return A
    try: return D + (A - D) / (1.0 + (x / C) ** B)
    except: return np.nan

def inv_rod_4pl(y, A, B, C, D):
    """Tính Nồng độ từ Tín hiệu"""
    try:
        if (A - D) == 0 or (y - D) == 0: return np.nan
        term = (A - D) / (y - D) - 1
        if term <= 0: return np.nan
        return C * (term ** (1/B))
    except: return np.nan

def parse_roche_xml(uploaded_file):
    try:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        test_name = "Unknown"
        for child in root.iter():
            if 'ContainerNameShort' in child.attrib:
                test_name = child.attrib['ContainerNameShort']
                break
        
        quant_tag = None
        for child in root.iter():
            if 'RodbardCurveParameters' in child.attrib:
                quant_tag = child
                break
        
        qual_tag = None
        for child in root.iter():
            if 'CutoffFNeg' in child.attrib:
                qual_tag = child
                break
        return test_name, quant_tag, qual_tag
    except Exception as e:
        st.error(f"Lỗi đọc file XML: {e}")
        return None, None, None

# ==============================================================================
# 3. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # IMPORT XML
    st.markdown("### 📂 Import Parameter File")
    uploaded_file = st.file_uploader("Upload Roche XML", type=['xml'])
    
    if uploaded_file is not None:
        name, quant_data, qual_data = parse_roche_xml(uploaded_file)
        if name:
            st.success(f"Đã tải xét nghiệm: **{name}**")
            if quant_data is not None:
                p_str = quant_data.attrib['RodbardCurveParameters']
                p_vals = [float(x) for x in p_str.split()]
                st.session_state.master_params = {'A': p_vals[0], 'C': p_vals[1], 'B': p_vals[2], 'D': p_vals[3]}
                st.toast("Đã cập nhật tham số Master Curve (4PL)", icon="✅")
            if qual_data is not None:
                attr = qual_data.attrib
                st.session_state.qual_params = {
                    'FNeg': float(attr.get('CutoffFNeg', 1)),
                    'FPos': float(attr.get('CutoffFPos', 0.65)),
                    'Const': float(attr.get('CutoffC', 0)),
                    'MinNeg': float(attr.get('MinSignalNegativeCalibration', 0)),
                    'MaxNeg': float(attr.get('MaxSignalNegativeCalibration', 99999)),
                    'MinPos': float(attr.get('MinSignalPositiveCalibration', 0)),
                    'MaxPos': float(attr.get('MaxSignalPositiveCalibration', 999999)),
                    'MinDiff': float(attr.get('MinAcceptableCalibratorSignalDifference', 0))
                }
                st.toast("Đã cập nhật tham số Cutoff", icon="✅")
    
    st.divider()
    app_mode = st.radio("Chọn Chức năng:", ["1. Định lượng (Quantitative)", "2. Định tính (Qualitative)", "3. Troubleshoot (Lịch sử)"])
    st.divider()
    
    # MANUAL EDIT
    if app_mode == "1. Định lượng (Quantitative)" or app_mode == "3. Troubleshoot (Lịch sử)":
        st.subheader("⚙️ Master Curve (4PL)")
        p = st.session_state.master_params
        mA = st.number_input("A", value=p['A'], format="%.0f")
        mB = st.number_input("B", value=p['B'], format="%.6f")
        mC = st.number_input("C", value=p['C'], format="%.4f")
        mD = st.number_input("D", value=p['D'], format="%.0f")
        st.session_state.master_params.update({'A': mA, 'B': mB, 'C': mC, 'D': mD})
    elif app_mode == "2. Định tính (Qualitative)":
        st.subheader("⚙️ Cutoff Params")
        qp = st.session_state.qual_params
        q_FNeg = st.number_input("Fac Neg", value=qp['FNeg'])
        q_FPos = st.number_input("Fac Pos", value=qp['FPos'])
        q_Const = st.number_input("Const", value=qp['Const'])
        st.session_state.qual_params.update({'FNeg': q_FNeg, 'FPos': q_FPos, 'Const': q_Const})

# ==============================================================================
# MODE 1: ĐỊNH LƯỢNG (QUANTITATIVE)
# ==============================================================================
if app_mode == "1. Định lượng (Quantitative)":
    st.title("🧪 Định lượng (4PL Recalibration)")
    
    col_in, col_out = st.columns([1, 1.5])
    
    with col_in:
        st.subheader("1. Recalibration")
        c1, c2 = st.columns(2)
        with c1:
            t1 = st.number_input("Target 1:", value=42.1)
            s1 = st.number_input("Signal 1:", value=583722.0)
        with c2:
            t2 = st.number_input("Target 2:", value=372.0)
            s2 = st.number_input("Signal 2:", value=288320.0)
            
        if st.button("🚀 Thực hiện Cal", type="primary"):
            p = st.session_state.master_params
            ms1 = rod_4pl(t1, **p)
            ms2 = rod_4pl(t2, **p)
            
            if (ms2 - ms1) != 0:
                slope = (s2 - s1) / (ms2 - ms1)
                intercept = s1 - slope * ms1
                st.session_state.quant_results = {'slope': slope, 'intercept': intercept, 't1': t1, 't2': t2, 's1': s1, 's2': s2}
            else:
                st.error("Lỗi tính toán: Mẫu số bằng 0")

    with col_out:
        if st.session_state.quant_results:
            res = st.session_state.quant_results
            p = st.session_state.master_params
            
            # KPI DISPLAY
            k1, k2, k3 = st.columns(3)
            k1.metric("Slope", f"{res['slope']:.4f}")
            k2.metric("Intercept", f"{res['intercept']:.0f}")
            
            # Đánh giá PASS/FAIL
            is_pass = 0.8 <= res['slope'] <= 1.2
            if is_pass:
                k3.success("✅ PASS")
            else:
                k3.error("❌ FAIL") # Hiển thị Fail nhưng vẫn tiếp tục vẽ bên dưới
            
            # --- VẼ BIỂU ĐỒ (CẬP NHẬT RANGE TỰ ĐỘNG) ---
            st.subheader("2. Biểu đồ Recalibration")
            
            # Tự động tìm Min/Max để vẽ cho đẹp
            # Lấy min của target, chia 5 để có khoảng hở bên trái
            min_x = min(res['t1'], res['t2']) / 5 
            if min_x <= 0: min_x = 0.01 # Tránh lỗi log(0)
            
            # Lấy max của target, nhân 5 để có khoảng hở bên phải
            max_x = max(res['t1'], res['t2']) * 5
            
            # Tạo dải X mới dựa trên dữ liệu thật
            x_plot = np.logspace(np.log10(min_x), np.log10(max_x), 200)
            
            y_master = [rod_4pl(x, **p) for x in x_plot]
            y_recal = [y * res['slope'] + res['intercept'] for y in y_master]
            
            fig = go.Figure()
            # Master Curve
            fig.add_trace(go.Scatter(x=x_plot, y=y_master, mode='lines', name='Master (Gốc)', line=dict(dash='dash', color='gray')))
            # Actual Curve
            line_color = 'blue' if is_pass else 'red'
            line_name = 'Hiện tại (OK)' if is_pass else 'Hiện tại (FAIL)'
            fig.add_trace(go.Scatter(x=x_plot, y=y_recal, mode='lines', name=line_name, line=dict(color=line_color, width=3)))
            # Points
            fig.add_trace(go.Scatter(x=[res['t1'], res['t2']], y=[res['s1'], res['s2']], mode='markers', name='Điểm Cal', marker=dict(size=12, color='black', symbol='x')))
            
            fig.update_layout(xaxis_type="log", yaxis_type="log", height=450, title="So sánh Master vs Thực tế")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- CÔNG CỤ TÍNH 2 CHIỀU (LUÔN HIỆN) ---
            st.divider()
            calc_type = st.radio("Chuyển đổi:", ["Signal ➔ Result", "Result ➔ Signal"], horizontal=True)
            
            if calc_type == "Signal ➔ Result":
                with st.form("calc_s2r"):
                    in_sig = st.number_input("Nhập Signal mẫu:", value=400000.0)
                    if st.form_submit_button("Tính Result"):
                        norm_sig = (in_sig - res['intercept']) / res['slope']
                        final_conc = inv_rod_4pl(norm_sig, **p)
                        st.success(f"Kết quả: **{final_conc:.4f}**")
                        # Vẽ điểm mẫu
                        fig.add_trace(go.Scatter(x=[final_conc], y=[in_sig], mode='markers', name='Mẫu', marker=dict(size=15, color='orange', symbol='star')))
                        st.plotly_chart(fig, use_container_width=True, key='chart_s2r')
            else:
                with st.form("calc_r2s"):
                    in_conc = st.number_input("Nhập Result mong muốn:", value=100.0)
                    if st.form_submit_button("Dự đoán Signal"):
                        master_sig = rod_4pl(in_conc, **p)
                        pred_sig = master_sig * res['slope'] + res['intercept']
                        st.info(f"Signal dự kiến: **{pred_sig:,.0f}**")

# ==============================================================================
# MODE 2: ĐỊNH TÍNH (QUALITATIVE)
# ==============================================================================
elif app_mode == "2. Định tính (Qualitative)":
    st.title("⚖️ Định tính (Cutoff & COI)")
    qp = st.session_state.qual_params
    
    col_in, col_out = st.columns([1, 1.5])
    
    with col_in:
        st.subheader("1. Xác lập Cutoff")
        sig_neg = st.number_input("Cal 1 (Neg):", value=2000.0)
        sig_pos = st.number_input("Cal 2 (Pos):", value=50000.0)
        
        if st.button("🚀 Tính Cutoff", type="primary"):
            msgs = []
            is_pass = True
            # QC Checks
            if not (qp['MinNeg'] <= sig_neg <= qp['MaxNeg']): is_pass = False; msgs.append(f"Neg ngoài dải ({qp['MinNeg']}-{qp['MaxNeg']})")
            if not (qp['MinPos'] <= sig_pos <= qp['MaxPos']): is_pass = False; msgs.append(f"Pos ngoài dải ({qp['MinPos']}-{qp['MaxPos']})")
            if (sig_pos - sig_neg) < qp['MinDiff']: is_pass = False; msgs.append(f"Diff quá nhỏ (<{qp['MinDiff']})")
            
            cutoff = (sig_neg * qp['FNeg']) + (sig_pos * qp['FPos']) + qp['Const']
            st.session_state.qual_results = {'cutoff': cutoff, 'is_pass': is_pass, 'msgs': msgs, 'sig_neg': sig_neg, 'sig_pos': sig_pos}

    with col_out:
        if st.session_state.qual_results:
            res = st.session_state.qual_results
            st.subheader("2. Kết quả & Biểu đồ")
            
            # Báo cáo Pass/Fail
            if res['is_pass']:
                st.success(f"✅ PASSED | Cutoff = {res['cutoff']:,.0f}")
            else:
                st.error(f"⛔ FAILED | Cutoff = {res['cutoff']:,.0f} (Invalid)")
                for m in res['msgs']: st.write(m)
            
            # --- VẼ BIỂU ĐỒ (LUÔN VẼ DÙ FAIL) ---
            # Để người dùng thấy trực quan tại sao Fail (ví dụ cột Neg quá cao)
            fig_bar = go.Figure()
            # Cột Neg
            color_neg = 'green' if (qp['MinNeg'] <= res['sig_neg'] <= qp['MaxNeg']) else 'red'
            fig_bar.add_trace(go.Bar(x=['Neg Cal'], y=[res['sig_neg']], marker_color=color_neg, name='Negative'))
            
            # Cột Cutoff
            fig_bar.add_trace(go.Bar(x=['Cutoff'], y=[res['cutoff']], marker_color='gray', name='Cutoff'))
            
            # Cột Pos
            color_pos = 'blue' if (qp['MinPos'] <= res['sig_pos'] <= qp['MaxPos']) else 'red'
            fig_bar.add_trace(go.Bar(x=['Pos Cal'], y=[res['sig_pos']], marker_color=color_pos, name='Positive'))
            
            # Vẽ các đường giới hạn (Min/Max) để dễ so sánh
            fig_bar.add_hline(y=qp['MaxNeg'], line_dash="dot", annotation_text="Max Neg", line_color="green")
            fig_bar.add_hline(y=qp['MinPos'], line_dash="dot", annotation_text="Min Pos", line_color="blue")
            
            fig_bar.update_layout(title="Trực quan hóa Tín hiệu Cal", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # --- CÔNG CỤ TÍNH MẪU (LUÔN HIỆN) ---
            st.divider()
            q_calc = st.radio("Tính toán:", ["Signal ➔ COI", "COI ➔ Signal"], horizontal=True)
            
            if q_calc == "Signal ➔ COI":
                with st.form("calc_coi"):
                    in_sig = st.number_input("Signal mẫu:", value=100000.0)
                    if st.form_submit_button("Tính COI"):
                        coi = in_sig / res['cutoff']
                        concl = "DƯƠNG TÍNH" if coi >= 1.0 else "ÂM TÍNH"
                        st.metric("COI", f"{coi:.2f}", concl)
                        # Vẽ điểm mẫu
                        fig_bar.add_trace(go.Scatter(x=['Mẫu'], y=[in_sig], mode='markers', marker=dict(size=15, color='orange', symbol='star')))
                        st.plotly_chart(fig_bar, use_container_width=True, key='qual_chart_upd')
            else:
                with st.form("calc_sig_q"):
                    in_coi = st.number_input("COI mong muốn:", value=1.0)
                    if st.form_submit_button("Dự đoán Signal"):
                        pred_sig = in_coi * res['cutoff']
                        st.info(f"Signal dự kiến: **{pred_sig:,.0f}**")

# ==============================================================================
# MODE 3: TROUBLESHOOT (LỊCH SỬ & VISUALIZATION)
# ==============================================================================
elif app_mode == "3. Troubleshoot (Lịch sử)":
    st.title("📈 Phân tích Xu hướng & So sánh Đường chuẩn")
    st.markdown("Theo dõi biến động Slope và trực quan hóa sự thay đổi hình dạng đường cong theo thời gian.")
    
    # Dữ liệu mẫu khởi tạo
    df_sample = pd.DataFrame([
        {"Date": "2023-12-01", "Target 1": 0.592, "Target 2": 19.0, "Signal 1": 4428, "Signal 2": 115877},
        {"Date": "2023-12-15", "Target 1": 0.592, "Target 2": 19.0, "Signal 1": 7336, "Signal 2": 117647},
    ])
    
    st.subheader("1. Dữ liệu Lịch sử Cal")
    edited_df = st.data_editor(df_sample, num_rows="dynamic", use_container_width=True)
    
    if st.button("🔍 Phân tích & Vẽ đồ thị", type="primary"):
        p = st.session_state.master_params
        
        # Danh sách kết quả để vẽ
        analysis_results = []
        
        # Biến để xác định Min/Max cho trục X của biểu đồ (Tránh bị vẽ ngắn/cụt)
        global_min_target = 99999
        global_max_target = 0
        
        # --- BƯỚC 1: TÍNH TOÁN SLOPE CHO TỪNG DÒNG ---
        for i, row in edited_df.iterrows():
            try:
                date_str = str(row['Date'])
                t1, t2 = float(row['Target 1']), float(row['Target 2'])
                s1, s2 = float(row['Signal 1']), float(row['Signal 2'])
                
                # Cập nhật min/max global để vẽ biểu đồ cho đẹp
                global_min_target = min(global_min_target, t1, t2)
                global_max_target = max(global_max_target, t1, t2)
                
                # Tính Master Signal
                m1 = rod_4pl(t1, **p)
                m2 = rod_4pl(t2, **p)
                
                # Tính Slope & Intercept cho ngày hôm đó
                if (m2 - m1) != 0:
                    slope = (s2 - s1) / (m2 - m1)
                    intercept = s1 - slope * m1
                    
                    analysis_results.append({
                        'Date': date_str,
                        'Slope': slope,
                        'Intercept': intercept,
                        'T1': t1, 'T2': t2,
                        'S1': s1, 'S2': s2
                    })
            except Exception as e:
                pass # Bỏ qua dòng lỗi
            
        # Chuyển thành DataFrame kết quả
        res_df = pd.DataFrame(analysis_results)
        
        if res_df.empty:
            st.error("Không có dữ liệu hợp lệ để phân tích.")
        else:
            st.divider()
            
            # --- BƯỚC 2: VẼ 2 BIỂU ĐỒ SONG SONG ---
            col_trend, col_overlay = st.columns(2)
            
            # --- BIỂU ĐỒ 1: XU HƯỚNG SLOPE (Trend Chart) ---
            with col_trend:
                st.subheader("A. Xu hướng Slope")
                st.caption("Theo dõi độ suy hao tín hiệu (Chuẩn = 1.0)")
                
                fig_trend = go.Figure()
                # Vùng an toàn
                fig_trend.add_hrect(y0=0.8, y1=1.2, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Safe Zone")
                # Đường Slope
                fig_trend.add_trace(go.Scatter(
                    x=res_df['Date'], y=res_df['Slope'],
                    mode='lines+markers+text',
                    text=[f"{s:.2f}" for s in res_df['Slope']],
                    textposition="top center",
                    name='Slope', line=dict(color='blue', width=2)
                ))
                fig_trend.update_layout(yaxis_title="Slope Factor", height=450)
                st.plotly_chart(fig_trend, use_container_width=True)

            # --- BIỂU ĐỒ 2: CHỒNG LỚP ĐƯỜNG CONG (Overlay Chart) ---
            with col_overlay:
                st.subheader("B. So sánh các Đường Cal")
                st.caption("Master (Nét đứt) vs Các lần chạy thực tế")
                
                fig_overlay = go.Figure()
                
                # 1. Tạo trục X mượt (Range động dựa trên min/max data)
                # Mở rộng biên trái phải một chút (chia 2, nhân 2)
                x_start = global_min_target / 5 if global_min_target > 0 else 0.01
                x_end = global_max_target * 5
                x_plot = np.logspace(np.log10(x_start), np.log10(x_end), 200)
                
                # 2. Vẽ Master Curve (Gốc) - Nằm dưới cùng
                y_master_base = [rod_4pl(x, **p) for x in x_plot]
                fig_overlay.add_trace(go.Scatter(
                    x=x_plot, y=y_master_base,
                    mode='lines', name='MASTER GỐC',
                    line=dict(color='black', dash='dash', width=2),
                    opacity=0.6
                ))
                
                # 3. Vẽ từng đường Cal lịch sử
                # Dùng phổ màu hoặc opacity để phân biệt
                for idx, row in res_df.iterrows():
                    # Tính đường cong của ngày hôm đó: y = y_master * slope + intercept
                    y_actual_curve = [y * row['Slope'] + row['Intercept'] for y in y_master_base]
                    
                    # Tên hiển thị trong chú thích
                    label = f"{row['Date']} (Slope: {row['Slope']:.2f})"
                    
                    # Vẽ đường cong
                    fig_overlay.add_trace(go.Scatter(
                        x=x_plot, y=y_actual_curve,
                        mode='lines', name=label
                    ))
                    
                    # Vẽ điểm Cal thực tế của ngày đó (để kiểm chứng độ khớp)
                    fig_overlay.add_trace(go.Scatter(
                        x=[row['T1'], row['T2']], y=[row['S1'], row['S2']],
                        mode='markers', showlegend=False,
                        marker=dict(size=8, symbol='circle')
                    ))
                
                fig_overlay.update_layout(
                    xaxis_type="log", yaxis_type="log",
                    xaxis_title="Nồng độ (Log)", yaxis_title="Tín hiệu (Log)",
                    height=450,
                    legend=dict(orientation="h", y=-0.2) # Đưa chú thích xuống dưới cho đỡ rối
                )
                st.plotly_chart(fig_overlay, use_container_width=True)
            
            # --- BƯỚC 3: BẢNG CHI TIẾT ---
            with st.expander("Xem bảng chi tiết tham số tính toán"):
                st.dataframe(res_df.style.format({
                    "Slope": "{:.4f}", "Intercept": "{:.2f}",
                    "S1": "{:.0f}", "S2": "{:.0f}"
                }))
