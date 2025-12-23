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
    """Đọc file XML và trích xuất tham số"""
    try:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        
        # Tìm tên xét nghiệm
        test_name = "Unknown"
        for child in root.iter():
            if 'ContainerNameShort' in child.attrib:
                test_name = child.attrib['ContainerNameShort']
                break

        # 1. Tìm thẻ ĐỊNH LƯỢNG (Quantitative)
        quant_tag = None
        for child in root.iter():
            if 'RodbardCurveParameters' in child.attrib:
                quant_tag = child
                break
        
        # 2. Tìm thẻ ĐỊNH TÍNH (Qualitative)
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
# 3. SIDEBAR: IMPORT & CẤU HÌNH
# ==============================================================================
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # --- MODULE IMPORT XML ---
    st.markdown("### 📂 Import Parameter File")
    uploaded_file = st.file_uploader("Upload Roche XML", type=['xml'])
    
    if uploaded_file is not None:
        name, quant_data, qual_data = parse_roche_xml(uploaded_file)
        
        if name:
            st.success(f"Đã tải xét nghiệm: **{name}**")
            
            # Xử lý dữ liệu định lượng
            if quant_data is not None:
                p_str = quant_data.attrib['RodbardCurveParameters']
                # Mapping Roche Order: A (Dose 0), C (IC50), B (Slope), D (Inf)
                p_vals = [float(x) for x in p_str.split()]
                st.session_state.master_params = {
                    'A': p_vals[0], 'C': p_vals[1], 'B': p_vals[2], 'D': p_vals[3]
                }
                st.toast("Đã cập nhật tham số Master Curve (4PL)", icon="✅")

            # Xử lý dữ liệu định tính
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
    
    # CHỌN CHẾ ĐỘ
    app_mode = st.radio(
        "Chọn Chức năng:",
        ["1. Định lượng (Quantitative)", "2. Định tính (Qualitative)", "3. Troubleshoot (Lịch sử)"]
    )
    
    st.divider()
    
    # HIỂN THỊ THAM SỐ (Cho phép sửa tay sau khi import)
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
                st.session_state.quant_results = {
                    'slope': slope, 'intercept': intercept,
                    't1': t1, 't2': t2, 's1': s1, 's2': s2
                }
                st.success("Recalibration OK!")
            else:
                st.error("Lỗi tính toán!")

    with col_out:
        if st.session_state.quant_results:
            res = st.session_state.quant_results
            p = st.session_state.master_params
            
            # KPI
            k1, k2, k3 = st.columns(3)
            k1.metric("Slope", f"{res['slope']:.4f}")
            k2.metric("Intercept", f"{res['intercept']:.0f}")
            status = "✅ PASS" if 0.8 <= res['slope'] <= 1.2 else "❌ FAIL"
            k3.metric("Status", status)
            
            # --- CÔNG CỤ TÍNH TOÁN 2 CHIỀU ---
            st.divider()
            st.subheader("2. Công cụ chuyển đổi (2 Chiều)")
            
            calc_type = st.radio("Chọn hướng tính toán:", 
                                 ["📡 Signal ➔ Result (Tính kết quả mẫu)", 
                                  "🧪 Result ➔ Signal (Dự đoán tín hiệu)"], 
                                 horizontal=True)
            
            if calc_type == "📡 Signal ➔ Result (Tính kết quả mẫu)":
                with st.form("calc_sig_to_res"):
                    in_sig = st.number_input("Nhập Signal mẫu:", value=400000.0)
                    if st.form_submit_button("Tính Result"):
                        # Quy trình: Signal Thô -> Chuẩn hóa (trừ nền/chia slope) -> Tra ngược Master
                        norm_sig = (in_sig - res['intercept']) / res['slope']
                        final_conc = inv_rod_4pl(norm_sig, **p)
                        
                        st.success(f"Kết quả: **{final_conc:.4f}**")
                        st.caption(f"(Tín hiệu đã chuẩn hóa về Master: {norm_sig:.0f})")
                        
            else: # Result -> Signal
                with st.form("calc_res_to_sig"):
                    in_conc = st.number_input("Nhập Result mong muốn:", value=100.0)
                    if st.form_submit_button("Dự đoán Signal"):
                        # Quy trình: Tra xuôi Master -> Biến đổi (nhân slope + nền) -> Signal Thô
                        master_sig = rod_4pl(in_conc, **p)
                        pred_sig = master_sig * res['slope'] + res['intercept']
                        
                        st.info(f"Tín hiệu dự kiến: **{pred_sig:,.0f}**")
                        st.caption(f"(Tín hiệu trên Master gốc: {master_sig:,.0f})")

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
            # Simple QC Checks
            if not (qp['MinNeg'] <= sig_neg <= qp['MaxNeg']): is_pass = False; msgs.append("Neg ngoài dải")
            if not (qp['MinPos'] <= sig_pos <= qp['MaxPos']): is_pass = False; msgs.append("Pos ngoài dải")
            if (sig_pos - sig_neg) < qp['MinDiff']: is_pass = False; msgs.append("Diff quá nhỏ")
            
            cutoff = (sig_neg * qp['FNeg']) + (sig_pos * qp['FPos']) + qp['Const']
            st.session_state.qual_results = {'cutoff': cutoff, 'is_pass': is_pass, 'msgs': msgs}

    with col_out:
        if st.session_state.qual_results:
            res = st.session_state.qual_results
            st.subheader("2. Kết quả")
            if res['is_pass']:
                st.success(f"Cutoff = {res['cutoff']:,.0f}")
                
                # --- TÍNH TOÁN 2 CHIỀU ---
                st.divider()
                st.subheader("3. Công cụ chuyển đổi")
                
                q_calc_type = st.radio("Hướng tính:", ["Signal ➔ COI", "COI ➔ Signal"], horizontal=True)
                
                if q_calc_type == "Signal ➔ COI":
                    with st.form("calc_coi"):
                        in_sig = st.number_input("Signal mẫu:", value=100000.0)
                        if st.form_submit_button("Tính COI"):
                            coi = in_sig / res['cutoff']
                            concl = "DƯƠNG TÍNH" if coi >= 1.0 else "ÂM TÍNH"
                            st.metric("COI", f"{coi:.2f}", concl)
                            
                else: # COI -> Signal
                    with st.form("calc_sig_q"):
                        in_coi = st.number_input("COI mong muốn:", value=1.0)
                        if st.form_submit_button("Tính Signal"):
                            # Signal = COI * Cutoff
                            pred_sig = in_coi * res['cutoff']
                            st.info(f"Tín hiệu tương ứng: **{pred_sig:,.0f}**")
                            
            else:
                st.error("Cal Failed")
                for m in res['msgs']: st.write(m)

# ==============================================================================
# MODE 3: TROUBLESHOOT
# ==============================================================================
elif app_mode == "3. Troubleshoot (Lịch sử)":
    st.title("📈 Phân tích Xu hướng (Trend)")
    st.info("Nhập dữ liệu lịch sử để vẽ biểu đồ.")
    
    # Dữ liệu demo
    df_sample = pd.DataFrame([
        {"Date": "2023-12-01", "Target 1": 42.1, "Target 2": 372.0, "Signal 1": 590000, "Signal 2": 295000},
        {"Date": "2023-12-15", "Target 1": 42.1, "Target 2": 372.0, "Signal 1": 583602, "Signal 2": 289073},
    ])
    edited_df = st.data_editor(df_sample, num_rows="dynamic", use_container_width=True)
    
    if st.button("🔍 Phân tích"):
        p = st.session_state.master_params
        res_list = []
        for i, row in edited_df.iterrows():
            try:
                t1, t2 = float(row['Target 1']), float(row['Target 2'])
                s1, s2 = float(row['Signal 1']), float(row['Signal 2'])
                m1, m2 = rod_4pl(t1, **p), rod_4pl(t2, **p)
                slope = (s2 - s1) / (m2 - m1)
                res_list.append({'Date': row['Date'], 'Slope': slope})
            except: pass
            
        rdf = pd.DataFrame(res_list)
        fig = go.Figure()
        fig.add_hrect(y0=0.8, y1=1.2, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_trace(go.Scatter(x=rdf['Date'], y=rdf['Slope'], mode='lines+markers', name='Slope'))
        st.plotly_chart(fig, use_container_width=True)
