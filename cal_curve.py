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
# MODE 3: TROUBLESHOOT (LỊCH SỬ & MÔ PHỎNG)
# ==============================================================================
elif app_mode == "3. Troubleshoot (Lịch sử)":
    st.title("📈 Phân tích Xu hướng & Mô phỏng")
    st.markdown("Theo dõi biến động Slope và đánh giá tác động lên kết quả bệnh nhân.")
    
    # 1. KHỞI TẠO STATE CHO MODE 3 (Để không bị mất khi bấm nút tính)
    if 'history_analysis' not in st.session_state:
        st.session_state.history_analysis = None

    # 2. DỮ LIỆU ĐẦU VÀO
    df_sample = pd.DataFrame([
        {"Date": "2023-12-01", "Target 1": 0.592, "Target 2": 19.0, "Signal 1": 4428, "Signal 2": 115877},
        {"Date": "2023-12-15", "Target 1": 0.592, "Target 2": 19.0, "Signal 1": 7336, "Signal 2": 117647},
        {"Date": "2023-12-30", "Target 1": 0.592, "Target 2": 19.0, "Signal 1": 3500, "Signal 2": 100000},
    ])
    
    st.subheader("1. Dữ liệu Lịch sử Cal")
    # data_editor tự giữ trạng thái nên không lo mất dữ liệu nhập
    edited_df = st.data_editor(df_sample, num_rows="dynamic", use_container_width=True)
    
    # 3. NÚT PHÂN TÍCH (Chỉ làm nhiệm vụ TÍNH và LƯU vào State)
    if st.button("🔍 Phân tích dữ liệu", type="primary"):
        p = st.session_state.master_params
        analysis_results = []
        global_min, global_max = 99999, 0
        
        for i, row in edited_df.iterrows():
            try:
                date_str = str(row['Date'])
                t1, t2 = float(row['Target 1']), float(row['Target 2'])
                s1, s2 = float(row['Signal 1']), float(row['Signal 2'])
                
                # Update range vẽ biểu đồ
                global_min = min(global_min, t1, t2)
                global_max = max(global_max, t1, t2)
                
                m1, m2 = rod_4pl(t1, **p), rod_4pl(t2, **p)
                
                if (m2 - m1) != 0:
                    slope = (s2 - s1) / (m2 - m1)
                    intercept = s1 - slope * m1
                    
                    analysis_results.append({
                        'Date': date_str, 'Slope': slope, 'Intercept': intercept,
                        'T1': t1, 'T2': t2, 'S1': s1, 'S2': s2
                    })
            except: pass
        
        # LƯU KẾT QUẢ VÀO SESSION STATE (QUAN TRỌNG)
        if analysis_results:
            st.session_state.history_analysis = {
                'results': analysis_results,
                'min_x': global_min,
                'max_x': global_max
            }
            st.success("Đã phân tích xong! Kéo xuống để xem kết quả.")
        else:
            st.error("Không có dữ liệu hợp lệ.")

    # 4. HIỂN THỊ KẾT QUẢ (Luôn hiển thị nếu State đã có dữ liệu)
    if st.session_state.history_analysis is not None:
        data = st.session_state.history_analysis
        res_list = data['results']
        res_df = pd.DataFrame(res_list)
        p = st.session_state.master_params
        
        st.divider()
        
        # --- A. BIỂU ĐỒ ---
        col_trend, col_overlay = st.columns(2)
        with col_trend:
            st.subheader("A. Xu hướng Slope")
            fig_trend = go.Figure()
            fig_trend.add_hrect(y0=0.8, y1=1.2, fillcolor="green", opacity=0.1, line_width=0)
            fig_trend.add_trace(go.Scatter(x=res_df['Date'], y=res_df['Slope'], mode='lines+markers+text', text=[f"{s:.2f}" for s in res_df['Slope']], textposition="top center", name='Slope'))
            fig_trend.update_layout(yaxis_title="Slope Factor", height=400)
            st.plotly_chart(fig_trend, use_container_width=True)

        with col_overlay:
            st.subheader("B. Overlay Đường cong")
            fig_overlay = go.Figure()
            x_start = data['min_x'] / 5 if data['min_x'] > 0 else 0.01
            x_end = data['max_x'] * 5
            x_plot = np.logspace(np.log10(x_start), np.log10(x_end), 200)
            
            # Master
            y_m_base = [rod_4pl(x, **p) for x in x_plot]
            fig_overlay.add_trace(go.Scatter(x=x_plot, y=y_m_base, mode='lines', name='MASTER', line=dict(color='black', dash='dash'), opacity=0.5))
            
            # History Curves
            for r in res_list:
                y_act = [y * r['Slope'] + r['Intercept'] for y in y_m_base]
                fig_overlay.add_trace(go.Scatter(x=x_plot, y=y_act, mode='lines', name=f"{r['Date']} (S:{r['Slope']:.2f})"))
            
            fig_overlay.update_layout(xaxis_type="log", yaxis_type="log", height=400, xaxis_title="Log Conc", yaxis_title="Log Signal")
            st.plotly_chart(fig_overlay, use_container_width=True)

        # --- B. MÔ PHỎNG TÁC ĐỘNG (Phần này sẽ KHÔNG bị reset nữa) ---
        st.divider()
        st.subheader("C. Mô phỏng & Chuyển đổi")
        st.markdown("Nhập giá trị để xem sự biến động kết quả qua các ngày.")
        
        sim_col1, sim_col2 = st.columns([1, 2])
        
        with sim_col1:
            # Dùng Form để gom nhóm input
            with st.form("sim_form"):
                sim_mode = st.radio("Chọn hướng:", ["Signal ➔ Result", "Result ➔ Signal"])
                sim_val = st.number_input("Nhập giá trị:", value=100000.0 if sim_mode == "Signal ➔ Result" else 5.0)
                
                # Nút tính nằm trong form
                calc_btn = st.form_submit_button("⚡ Tính toán")
        
        with sim_col2:
            if calc_btn: # Khi bấm nút này, code chạy lại nhưng history_analysis vẫn còn trong session_state
                sim_res_list = []
                input_lbl = "Signal" if sim_mode == "Signal ➔ Result" else "Result"
                output_lbl = "Result" if sim_mode == "Signal ➔ Result" else "Signal"
                
                for r in res_list:
                    out_val = np.nan
                    if sim_mode == "Signal ➔ Result":
                        norm = (sim_val - r['Intercept']) / r['Slope']
                        out_val = inv_rod_4pl(norm, **p)
                    else:
                        m_sig = rod_4pl(sim_val, **p)
                        out_val = m_sig * r['Slope'] + r['Intercept']
                    
                    sim_res_list.append({
                        "Date": r['Date'],
                        "Slope": r['Slope'],
                        output_lbl: out_val
                    })
                
                # Hiển thị kết quả
                df_sim = pd.DataFrame(sim_res_list)
                st.dataframe(df_sim.style.format({
                    "Slope": "{:.4f}",
                    output_lbl: "{:.4f}" if sim_mode == "Signal ➔ Result" else "{:,.0f}"
                }), use_container_width=True)
                
                # Vẽ biểu đồ biến động
                if not df_sim[output_lbl].isna().all():
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(
                        x=df_sim['Date'], y=df_sim[output_lbl],
                        mode='lines+markers+text',
                        text=[f"{v:.2f}" if sim_mode == "Signal ➔ Result" else f"{v:.0f}" for v in df_sim[output_lbl]],
                        textposition="top center",
                        line=dict(color='orange', width=2, dash='dot')
                    ))
                    
                    # Tính CV%
                    vals = df_sim[output_lbl].dropna()
                    if len(vals) > 0:
                        cv = (np.std(vals) / np.mean(vals)) * 100
                        st.caption(f"Độ biến thiên (CV%): **{cv:.2f}%**")
                        
                    fig_sim.update_layout(title=f"Biến động {output_lbl}", height=300)
                    st.plotly_chart(fig_sim, use_container_width=True)
