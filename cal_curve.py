import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xml.etree.ElementTree as ET

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Roche Cal Expert Ultra", layout="wide", page_icon="🧬")

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

if 'chem_ref_params' not in st.session_state: st.session_state.chem_ref_params = None
if 'quant_results' not in st.session_state: st.session_state.quant_results = None
if 'qual_results' not in st.session_state: st.session_state.qual_results = None
if 'history_analysis' not in st.session_state: st.session_state.history_analysis = None
if 'chem_linear_res' not in st.session_state: st.session_state.chem_linear_res = None
if 'chem_multipoint' not in st.session_state: st.session_state.chem_multipoint = None

# ==============================================================================
# 2. HÀM TOÁN HỌC CỐT LÕI
# ==============================================================================
# --- MIỄN DỊCH (4PL) ---
def rod_4pl(x, A, B, C, D):
    if x < 0: return A
    try: return D + (A - D) / (1.0 + (x / C) ** B)
    except: return np.nan

def inv_rod_4pl(y, A, B, C, D):
    try:
        if (A - D) == 0 or (y - D) == 0: return np.nan
        term = (A - D) / (y - D) - 1
        if term <= 0: return np.nan
        return C * (term ** (1/B))
    except: return np.nan

# [cite_start]--- SINH HÓA (LINEAR & LINE GRAPH) --- [cite: 317, 318, 354, 475]
def calc_k_factor(conc_std, conc_blank, abs_std, abs_blank):
    if (abs_std - abs_blank) == 0: return 0
    return (conc_std - conc_blank) / (abs_std - abs_blank)

def calc_linear_conc(abs_sample, k_factor, abs_blank, conc_blank):
    return k_factor * (abs_sample - abs_blank) + conc_blank

def calc_line_graph(abs_sample, cal_points):
    [cite_start]# [cite: 833, 881]
    points = sorted(cal_points, key=lambda k: k['abs'])
    if abs_sample <= points[0]['abs']:
        p1, p2 = points[0], points[1]
    elif abs_sample >= points[-1]['abs']:
        p1, p2 = points[-2], points[-1]
    else:
        for i in range(len(points) - 1):
            if points[i]['abs'] <= abs_sample <= points[i+1]['abs']:
                p1, p2 = points[i], points[i+1]
                break
    
    if (p2['abs'] - p1['abs']) == 0: return 0
    k_interval = (p2['conc'] - p1['conc']) / (p2['abs'] - p1['abs'])
    return k_interval * (abs_sample - p1['abs']) + p1['conc']

# ==============================================================================
# 3. XỬ LÝ XML (Parser Đa Năng - Cập nhật mới)
# ==============================================================================
def parse_roche_xml(uploaded_file):
    """
    Parser thông minh: Quét attribute thay vì fix cứng tên thẻ.
    Hỗ trợ: e801, e601, e602, c501, c503...
    """
    try:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        
        # 1. Tìm Tên xét nghiệm (Quét toàn bộ cây)
        test_name = "Unknown"
        # Ưu tiên tìm ContainerNameShort (e801), nếu không có tìm ApplicationCode (e602/c501)
        for elem in root.iter():
            if 'ContainerNameShort' in elem.attrib:
                test_name = elem.attrib['ContainerNameShort']
                break
            if 'ApplicationCode' in elem.attrib and test_name == "Unknown":
                test_name = f"AppCode {elem.attrib['ApplicationCode']}"

        # 2. Quét tìm Dữ liệu (Dựa trên Attribute đặc trưng)
        found_data = None
        
        # Quét toàn bộ các thẻ trong XML
        for elem in root.iter():
            
            # --- TRƯỜNG HỢP A: ĐỊNH LƯỢNG 4PL (Rodbard) ---
            # Dấu hiệu: Có thuộc tính 'RodbardCurveParameters'
            if 'RodbardCurveParameters' in elem.attrib:
                p_str = elem.attrib['RodbardCurveParameters']
                # Roche Format: "A C B D" (Space separated)
                # e602 Example: "168512 34.9558 0.955072 -16193.5"
                try:
                    p_vals = [float(x) for x in p_str.split()]
                    # Mapping chuẩn Roche thường là: Val[0]=A, Val[1]=C, Val[2]=B, Val[3]=D
                    found_data = {
                        "type": "immuno_quant", 
                        "name": test_name, 
                        "params": {'A': p_vals[0], 'C': p_vals[1], 'B': p_vals[2], 'D': p_vals[3]}
                    }
                except:
                    continue # Bỏ qua nếu lỗi format

            # --- TRƯỜNG HỢP B: ĐỊNH TÍNH (Cutoff) ---
            # Dấu hiệu: Có thuộc tính 'CutoffFNeg'
            elif 'CutoffFNeg' in elem.attrib:
                attr = elem.attrib
                found_data = {
                    "type": "immuno_qual", "name": test_name,
                    "params": {
                        'FNeg': float(attr.get('CutoffFNeg', 1)),
                        'FPos': float(attr.get('CutoffFPos', 0.65)),
                        'Const': float(attr.get('CutoffC', 0)),
                        'MinNeg': float(attr.get('MinSignalNegativeCalibration', 0)),
                        'MaxNeg': float(attr.get('MaxSignalNegativeCalibration', 99999)),
                        'MinPos': float(attr.get('MinSignalPositiveCalibration', 0)),
                        'MaxPos': float(attr.get('MaxSignalPositiveCalibration', 999999)),
                        'MinDiff': float(attr.get('MinAcceptableCalibratorSignalDifference', 0))
                    }
                }

        # --- TRƯỜNG HỢP C: SINH HÓA (Linear Pairs) ---
        # Dấu hiệu: Có các thẻ con ContainerReagentPair
        if found_data is None:
            pairs = []
            for elem in root.iter():
                if 'SxLot' in elem.attrib and 'CxLot' in elem.attrib:
                    sx = float(elem.get("SxLot"))
                    cx = float(elem.get("CxLot"))
                    pairs.append({"conc": cx, "abs": sx})
            
            if pairs:
                pairs.sort(key=lambda x: x['conc'])
                found_data = {"type": "chem_linear", "name": test_name, "points": pairs}

        return found_data

    except Exception as e:
        st.error(f"Lỗi đọc XML: {e}")
        return None

# ==============================================================================
# 4. GIAO DIỆN CHÍNH (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    st.markdown("### 📂 Nhập File Tham Số")
    uploaded_file = st.file_uploader("Upload Roche XML (e801, e602, c501...)", type=['xml'])
    
    if uploaded_file is not None:
        parsed = parse_roche_xml(uploaded_file)
        if parsed:
            st.success(f"Đã nhận diện: **{parsed['name']}**")
            
            if parsed['type'] == 'immuno_quant':
                st.session_state.master_params = parsed['params']
                st.toast("Đã cập nhật 4PL Parameters (e602/e801)", icon="✅")
                
            elif parsed['type'] == 'immuno_qual':
                st.session_state.qual_params = parsed['params']
                st.toast("Đã cập nhật Cutoff Parameters", icon="✅")
                
            elif parsed['type'] == 'chem_linear':
                st.session_state.chem_ref_params = parsed
                pts = parsed['points']
                if len(pts) >= 2:
                    k_ref = calc_k_factor(pts[1]['conc'], pts[0]['conc'], pts[1]['abs'], pts[0]['abs']) if (pts[1]['abs'] - pts[0]['abs']) != 0 else 0
                    st.session_state.chem_ref_params['k_ref'] = k_ref
                st.toast("Đã cập nhật Biochem Reference", icon="✅")
        else:
            st.error("Không tìm thấy dữ liệu hợp lệ trong XML!")

    st.divider()
    app_mode = st.radio("Chọn Chế độ:", 
                        ["1. Định lượng (Immuno 4PL)", 
                         "2. Định tính (Immuno Cutoff)", 
                         "3. Troubleshoot (Trend Analysis)",
                         "4. Sinh hóa (Photometric)"])
    
    st.divider()
    
    # Hiển thị tham số hiện tại để sửa tay
    if app_mode in ["1. Định lượng (Immuno 4PL)", "3. Troubleshoot (Trend Analysis)"]:
        st.caption("⚙️ Master Curve (4PL)")
        p = st.session_state.master_params
        for k in ['A', 'B', 'C', 'D']:
            st.session_state.master_params[k] = st.number_input(k, value=p[k], format="%.6f" if k=='B' else "%.2f")
            
    elif app_mode == "2. Định tính (Immuno Cutoff)":
        st.caption("⚙️ Cutoff Params")
        qp = st.session_state.qual_params
        for k in ['FNeg', 'FPos', 'Const']:
            st.session_state.qual_params[k] = st.number_input(k, value=qp[k])

# ==============================================================================
# MODE 1: ĐỊNH LƯỢNG (4PL)
# ==============================================================================
if app_mode == "1. Định lượng (Immuno 4PL)":
    st.title("🧪 Định lượng (4PL Recalibration)")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("1. Nhập Calibrator")
        c1, c2 = st.columns(2)
        with c1:
            t1 = st.number_input("Target 1", value=42.1)
            s1 = st.number_input("Signal 1", value=583722.0)
        with c2:
            t2 = st.number_input("Target 2", value=372.0)
            s2 = st.number_input("Signal 2", value=288320.0)
            
        if st.button("🚀 Thực hiện Cal", type="primary"):
            p = st.session_state.master_params
            m1, m2 = rod_4pl(t1, **p), rod_4pl(t2, **p)
            if (m2 - m1) != 0:
                slope = (s2 - s1) / (m2 - m1)
                intercept = s1 - slope * m1
                st.session_state.quant_results = {'slope': slope, 'intercept': intercept, 't1': t1, 't2': t2, 's1': s1, 's2': s2}
            else: st.error("Lỗi: Không thể tính toán (Mẫu số = 0)")

    with col2:
        if st.session_state.quant_results:
            res = st.session_state.quant_results
            p = st.session_state.master_params
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Slope", f"{res['slope']:.4f}")
            k2.metric("Intercept", f"{res['intercept']:.0f}")
            if 0.8 <= res['slope'] <= 1.2:
                k3.success("PASS")
            else:
                k3.error("FAIL")
            
            min_x = min(res['t1'], res['t2']) / 5 if min(res['t1'], res['t2']) > 0 else 0.01
            max_x = max(res['t1'], res['t2']) * 5
            x_plot = np.logspace(np.log10(min_x), np.log10(max_x), 200)
            y_m = [rod_4pl(x, **p) for x in x_plot]
            y_r = [y * res['slope'] + res['intercept'] for y in y_m]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_plot, y=y_m, mode='lines', name='Master', line=dict(dash='dash', color='gray')))
            fig.add_trace(go.Scatter(x=x_plot, y=y_r, mode='lines', name='Actual', line=dict(color='blue' if 0.8<=res['slope']<=1.2 else 'red')))
            fig.add_trace(go.Scatter(x=[res['t1'], res['t2']], y=[res['s1'], res['s2']], mode='markers', name='Points', marker=dict(color='black', symbol='x', size=10)))
            fig.update_layout(xaxis_type="log", yaxis_type="log", height=400, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            ctype = st.radio("Chuyển đổi:", ["Signal ➔ Result", "Result ➔ Signal"], horizontal=True)
            if ctype == "Signal ➔ Result":
                val = st.number_input("Signal:", value=400000.0)
                if st.button("Tính"):
                    conc = inv_rod_4pl((val - res['intercept']) / res['slope'], **p)
                    st.success(f"Kết quả: **{conc:.4f}**")
            else:
                val = st.number_input("Result:", value=100.0)
                if st.button("Tính"):
                    sig = rod_4pl(val, **p) * res['slope'] + res['intercept']
                    st.info(f"Signal: **{sig:,.0f}**")

# ==============================================================================
# MODE 2: ĐỊNH TÍNH (CUTOFF)
# ==============================================================================
elif app_mode == "2. Định tính (Immuno Cutoff)":
    st.title("⚖️ Định tính (Cutoff & COI)")
    qp = st.session_state.qual_params
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        s_neg = st.number_input("Cal Neg", value=2000.0)
        s_pos = st.number_input("Cal Pos", value=50000.0)
        if st.button("🚀 Tính Cutoff", type="primary"):
            msgs = []
            valid = True
            if not (qp['MinNeg'] <= s_neg <= qp['MaxNeg']): valid=False; msgs.append("Neg ngoài dải")
            if not (qp['MinPos'] <= s_pos <= qp['MaxPos']): valid=False; msgs.append("Pos ngoài dải")
            if (s_pos - s_neg) < qp['MinDiff']: valid=False; msgs.append("Diff quá nhỏ")
            
            cutoff = s_neg * qp['FNeg'] + s_pos * qp['FPos'] + qp['Const']
            st.session_state.qual_results = {'cutoff': cutoff, 'valid': valid, 'msgs': msgs, 'neg': s_neg, 'pos': s_pos}

    with col2:
        if st.session_state.qual_results:
            res = st.session_state.qual_results
            if res['valid']: st.success(f"PASS | Cutoff = {res['cutoff']:,.0f}")
            else: st.error(f"FAIL | {res['cutoff']:,.0f}"); [st.write(m) for m in res['msgs']]
            
            fig = go.Figure(data=[
                go.Bar(x=['Neg', 'Cutoff', 'Pos'], y=[res['neg'], res['cutoff'], res['pos']], marker_color=['green', 'gray', 'blue'])
            ])
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            sig_in = st.number_input("Signal mẫu:", value=100000.0)
            if st.button("Tính COI"):
                coi = sig_in / res['cutoff']
                st.metric("COI", f"{coi:.2f}", "DƯƠNG" if coi>=1 else "ÂM")

# ==============================================================================
# MODE 3: TROUBLESHOOT & SHAPE ANALYSIS
# ==============================================================================
elif app_mode == "3. Troubleshoot (Trend Analysis)":
    st.title("📈 Phân tích Hình học Đường chuẩn")
    st.markdown("Đánh giá hình dạng đường cong để tìm nguyên nhân gốc rễ.")
    
    col_in1, col_in2 = st.columns([1, 2])
    with col_in1:
        st.subheader("Thông số Cal hiện tại")
        t1 = st.number_input("Target 1", value=42.1)
        s1 = st.number_input("Signal 1 (Đo được)", value=583722.0)
        t2 = st.number_input("Target 2", value=372.0)
        s2 = st.number_input("Signal 2 (Đo được)", value=288320.0) 
        
    with col_in2:
        st.subheader("Phân tích Hình dạng")
        if st.button("🔍 Phân tích Hình học", type="primary"):
            p = st.session_state.master_params
            m1 = rod_4pl(t1, **p)
            m2 = rod_4pl(t2, **p)
            dev1 = ((s1 - m1) / m1) * 100
            dev2 = ((s2 - m2) / m2) * 100
            slope = (s2 - s1) / (m2 - m1) if (m2 - m1) != 0 else 0
            
            shape_type = "Bình thường"; color = "green"; advice = "Hệ thống ổn định."
            if (dev1 * dev2 < 0) and (abs(dev1 - dev2) > 10): 
                shape_type = "❌ MÉO MÓ / CẮT CHÉO"; color = "red"; advice = "Thao tác sai (bọt khí, lẫn mẫu)."
            elif abs(dev1 - dev2) < 5 and abs(dev1) > 10:
                shape_type = "⚠️ TỊNH TIẾN"; color = "orange"; advice = "Kiểm tra Nước rửa, Cuvette, Nhiễm bẩn."
            elif abs(dev1) < 5 and abs(dev2) > 10:
                if slope < 1: shape_type = "📉 XOAY XUỐNG"; color = "blue"; advice = "Già hóa thuốc thử/đèn."
                else: shape_type = "📈 XOAY LÊN"; color = "orange"; advice = "Thuốc thử bị cô đặc/bay hơi."

            st.markdown(f"### Kết luận: :{color}[{shape_type}]")
            st.info(f"💡 **Gợi ý:** {advice}")
            st.write(f"- Lệch Cal 1: **{dev1:+.1f}%** | Lệch Cal 2: **{dev2:+.1f}%** | Slope: **{slope:.4f}**")
            
            x_plot = np.logspace(np.log10(min(t1,t2)/5), np.log10(max(t1,t2)*5), 200)
            y_master = [rod_4pl(x, **p) for x in x_plot]
            intercept = s1 - slope * m1
            y_actual = [y * slope + intercept for y in y_master]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_plot, y=y_master, mode='lines', name='Master', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=x_plot, y=y_actual, mode='lines', name='Actual', line=dict(color=color, width=3)))
            fig.add_trace(go.Scatter(x=[t1, t2], y=[s1, s2], mode='markers', name='Points', marker=dict(color='black', size=10)))
            fig.update_layout(xaxis_type="log", yaxis_type="log", height=450)
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# MODE 4: SINH HÓA (PHOTOMETRIC)
# ==============================================================================
elif app_mode == "4. Sinh hóa (Photometric)":
    st.title("⚗️ Sinh hóa (Linear & Line Graph)")
    chem_type = st.selectbox("Thuật toán:", ["Linear 2-Point", "Line Graph (Multipoint)"])
    
    if chem_type == "Linear 2-Point":
        c1, c2 = st.columns(2)
        with c1:
            cb = st.number_input("Conc Std 1 (Cb)", value=0.0)
            ab = st.number_input("Abs Std 1 (Ab)", value=0.0036, format="%.4f")
        with c2:
            cn = st.number_input("Conc Std 2 (Cn)", value=10.8)
            an = st.number_input("Abs Std 2 (An)", value=0.8739, format="%.4f")
            
        if st.button("🚀 Tính K-Factor"):
            k = calc_k_factor(cn, cb, an, ab)
            st.session_state.chem_linear_res = {'k': k, 'cb': cb, 'ab': ab, 'cn': cn, 'an': an}
            
        if st.session_state.chem_linear_res:
            res = st.session_state.chem_linear_res
            st.divider()
            col_k, col_ref = st.columns(2)
            col_k.metric("K-Factor", f"{res['k']:.2f}")
            
            if st.session_state.chem_ref_params:
                ref_k = st.session_state.chem_ref_params.get('k_ref', 0)
                if ref_k != 0:
                    diff = ((res['k'] - ref_k)/ref_k)*100
                    col_ref.metric("Factory K", f"{ref_k:.2f}", f"{diff:.1f}%")
                    if abs(diff) > 20: st.warning("⚠️ Lệch > 20% so với XML")

            x_plt = np.linspace(0, res['cn']*1.5, 50)
            y_plt = [res['ab'] + (1/res['k'] if res['k']!=0 else 0) * (x - res['cb']) for x in x_plt]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_plt, y=y_plt, mode='lines', name='Linear Fit'))
            fig.add_trace(go.Scatter(x=[res['cb'], res['cn']], y=[res['ab'], res['an']], mode='markers', marker=dict(color='red', size=10)))
            st.plotly_chart(fig, use_container_width=True)
            
            with st.form("chem_l_calc"):
                s_abs = st.number_input("Abs Mẫu:", value=0.5000, format="%.4f")
                if st.form_submit_button("Tính Nồng độ"):
                    c_res = calc_linear_conc(s_abs, res['k'], res['ab'], res['cb'])
                    st.success(f"Kết quả: **{c_res:.2f}**")

    elif chem_type == "Line Graph (Multipoint)":
        st.write("Nhập bảng Calibrator:")
        df_pts = pd.DataFrame({"Conc": [0.0, 10.0, 50.0], "Abs": [0.005, 0.050, 0.220]})
        edt = st.data_editor(df_pts, num_rows="dynamic")
        if st.button("Lưu dữ liệu"):
            pts = [{"conc": float(r['Conc']), "abs": float(r['Abs'])} for i,r in edt.iterrows()]
            st.session_state.chem_multipoint = pts
            st.success("Đã lưu!")
            
        if st.session_state.chem_multipoint:
            pts = sorted(st.session_state.chem_multipoint, key=lambda x: x['conc'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[p['conc'] for p in pts], y=[p['abs'] for p in pts], mode='lines+markers'))
            st.plotly_chart(fig, use_container_width=True)
            s_abs = st.number_input("Abs Mẫu:", value=0.1)
            if st.button("Tính Line Graph"):
                res = calc_line_graph(s_abs, pts)
                st.success(f"Kết quả: **{res:.2f}**")
