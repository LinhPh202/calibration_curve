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
    st.title("📈 Phân tích Xu hướng & Mô phỏng Tác động")
    st.markdown("Theo dõi biến động Slope và đánh giá tác động lên kết quả bệnh nhân.")
    
    # Dữ liệu mẫu
    df_sample = pd.DataFrame([
        {"Date": "2023-12-01", "Target 1": 0.592, "Target 2": 19.0, "Signal 1": 4428, "Signal 2": 115877},
        {"Date": "2023-12-15", "Target 1": 0.592, "Target 2": 19.0, "Signal 1": 7336, "Signal 2": 117647},
        {"Date": "2023-12-30", "Target 1": 0.592, "Target 2": 19.0, "Signal 1": 3500, "Signal 2": 100000},
    ])
    
    st.subheader("1. Dữ liệu Lịch sử Cal")
    edited_df = st.data_editor(df_sample, num_rows="dynamic", use_container_width=True)
    
    if st.button("🔍 Phân tích & Mô phỏng", type="primary"):
        p = st.session_state.master_params
        
        # 1. TÍNH TOÁN SLOPE LỊCH SỬ
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
            
        res_df = pd.DataFrame(analysis_results)
        
        if not res_df.empty:
            st.divider()
            
            # --- 2. BIỂU ĐỒ TREND & OVERLAY (Giữ nguyên như cũ) ---
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
                x_start = global_min / 5 if global_min > 0 else 0.01
                x_end = global_max * 5
                x_plot = np.logspace(np.log10(x_start), np.log10(x_end), 200)
                
                # Master
                y_m_base = [rod_4pl(x, **p) for x in x_plot]
                fig_overlay.add_trace(go.Scatter(x=x_plot, y=y_m_base, mode='lines', name='MASTER', line=dict(color='black', dash='dash'), opacity=0.5))
                
                # History Curves
                for r in analysis_results:
                    y_act = [y * r['Slope'] + r['Intercept'] for y in y_m_base]
                    fig_overlay.add_trace(go.Scatter(x=x_plot, y=y_act, mode='lines', name=f"{r['Date']} (S:{r['Slope']:.2f})"))
                
                fig_overlay.update_layout(xaxis_type="log", yaxis_type="log", height=400, xaxis_title="Log Conc", yaxis_title="Log Signal")
                st.plotly_chart(fig_overlay, use_container_width=True)

            # --- 3. MÔ PHỎNG TÁC ĐỘNG (TÍNH NĂNG MỚI BẠN YÊU CẦU) ---
            st.divider()
            st.subheader("C. Mô phỏng & Chuyển đổi (Simulation)")
            st.markdown("Giả lập: Nếu chạy cùng một mẫu vào các ngày khác nhau, kết quả sẽ thay đổi thế nào?")
            
            sim_col1, sim_col2 = st.columns([1, 2])
            
            with sim_col1:
                sim_mode = st.radio("Chọn hướng tính:", ["Signal ➔ Result", "Result ➔ Signal"], horizontal=True)
                
                sim_input_val = 0.0
                if sim_mode == "Signal ➔ Result":
                    sim_input_val = st.number_input("Nhập Signal cố định (Ví dụ QC):", value=100000.0)
                    input_label = "Signal Input"
                    output_label = "Result Output"
                else:
                    sim_input_val = st.number_input("Nhập Result cố định (Ví dụ 5.0):", value=5.0)
                    input_label = "Result Input"
                    output_label = "Signal Output"
            
            with sim_col2:
                # Thực hiện tính toán hàng loạt cho tất cả các ngày
                sim_results = []
                
                for r in analysis_results:
                    val_out = np.nan
                    
                    if sim_mode == "Signal ➔ Result":
                        # Signal -> Norm -> Result
                        norm = (sim_input_val - r['Intercept']) / r['Slope']
                        val_out = inv_rod_4pl(norm, **p)
                    else:
                        # Result -> Master Sig -> Raw Sig
                        m_sig = rod_4pl(sim_input_val, **p)
                        val_out = m_sig * r['Slope'] + r['Intercept']
                    
                    sim_results.append({
                        "Date": r['Date'],
                        "Slope": r['Slope'],
                        input_label: sim_input_val,
                        output_label: val_out
                    })
                
                # Hiển thị bảng kết quả
                df_sim = pd.DataFrame(sim_results)
                
                # Format hiển thị cho đẹp
                st.dataframe(df_sim.style.format({
                    "Slope": "{:.4f}",
                    output_label: "{:.4f}" if sim_mode == "Signal ➔ Result" else "{:,.0f}"
                }), use_container_width=True)
                
                # Vẽ biểu đồ biến động kết quả
                if not df_sim[output_label].isna().all():
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(
                        x=df_sim['Date'], y=df_sim[output_label],
                        mode='lines+markers+text',
                        text=[f"{v:.2f}" if sim_mode == "Signal ➔ Result" else f"{v:.0f}" for v in df_sim[output_label]],
                        textposition="top center",
                        name='Simulated Value',
                        line=dict(color='orange', width=3, dash='dot')
                    ))
                    
                    # Tính % biến thiên (CV%)
                    vals = df_sim[output_label].dropna()
                    if len(vals) > 0:
                        avg = np.mean(vals)
                        cv = (np.std(vals) / avg) * 100
                        title_chart = f"Biến động {output_label} theo thời gian (CV%: {cv:.2f}%)"
                    else:
                        title_chart = "Biến động kết quả"
                        
                    fig_sim.update_layout(title=title_chart, height=350, yaxis_title=output_label)
                    st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.warning("Không có dữ liệu hợp lệ để phân tích.")
