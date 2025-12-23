import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Roche Cal Expert Pro", layout="wide", page_icon="🧪")

# ==============================================================================
# 1. QUẢN LÝ SESSION STATE (KHỞI TẠO DỮ LIỆU MẶC ĐỊNH)
# ==============================================================================

# A. Tham số Master Curve (Cho Định lượng & Troubleshoot) - Mặc định: Anti-TPO
if 'master_params' not in st.session_state:
    st.session_state.master_params = {'A': 876721.0, 'B': 0.762881, 'C': 175.289, 'D': -1315.11}

# B. Tham số Định tính (Cho Mode 2) - Mặc định: AHBCIGM
if 'qual_params' not in st.session_state:
    st.session_state.qual_params = {
        'FNeg': 1.0, 'FPos': 0.65, 'Const': 0.0,
        'MinNeg': 400.0, 'MaxNeg': 3500.0,
        'MinPos': 18000.0, 'MaxPos': 130000.0,
        'MinDiff': 16000.0
    }

# C. Lưu kết quả tính toán hiện tại
if 'quant_results' not in st.session_state: st.session_state.quant_results = None
if 'qual_results' not in st.session_state: st.session_state.qual_results = None

# ==============================================================================
# 2. HÀM TOÁN HỌC CỐT LÕI
# ==============================================================================
def rod_4pl(x, A, B, C, D):
    """Tính Tín hiệu từ Nồng độ (Master Curve)"""
    if x < 0: return A
    try: return D + (A - D) / (1.0 + (x / C) ** B)
    except: return np.nan

def inv_rod_4pl(y, A, B, C, D):
    """Tính Nồng độ từ Tín hiệu (Inverse 4PL)"""
    try:
        if (A - D) == 0 or (y - D) == 0: return np.nan
        term = (A - D) / (y - D) - 1
        if term <= 0: return np.nan
        return C * (term ** (1/B))
    except: return np.nan

# ==============================================================================
# 3. SIDEBAR: THANH ĐIỀU HƯỚNG & CẤU HÌNH
# ==============================================================================
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # CHỌN CHẾ ĐỘ
    app_mode = st.radio(
        "Chọn Chức năng:",
        ["1. Định lượng (Quantitative)", "2. Định tính (Qualitative)", "3. Troubleshoot (Lịch sử)"],
        captions=["Recalibration 4PL", "Cutoff & COI", "Trend Analysis"]
    )
    
    st.divider()
    
    # MENU CẤU HÌNH (THAY ĐỔI THEO MODE)
    if app_mode == "1. Định lượng (Quantitative)" or app_mode == "3. Troubleshoot (Lịch sử)":
        st.subheader("⚙️ Master Curve Parameters")
        st.caption("Nhập từ XML/Barcode (Anti-TPO...)")
        
        # Nhập và cập nhật ngay vào session state
        mA = st.number_input("A (Sig @0)", value=st.session_state.master_params['A'], format="%.0f")
        mB = st.number_input("B (Slope)", value=st.session_state.master_params['B'], format="%.6f")
        mC = st.number_input("C (IC50)", value=st.session_state.master_params['C'], format="%.4f")
        mD = st.number_input("D (Sig @Inf)", value=st.session_state.master_params['D'], format="%.0f")
        
        st.session_state.master_params.update({'A': mA, 'B': mB, 'C': mC, 'D': mD})
        
    elif app_mode == "2. Định tính (Qualitative)":
        st.subheader("⚙️ Cutoff Parameters")
        st.caption("Nhập từ XML (HBsAg, HCV...)")
        
        q_FNeg = st.number_input("Factor Neg", value=st.session_state.qual_params['FNeg'])
        q_FPos = st.number_input("Factor Pos", value=st.session_state.qual_params['FPos'])
        q_Const = st.number_input("Constant", value=st.session_state.qual_params['Const'])
        
        with st.expander("Giới hạn QC (Pass/Fail)"):
            q_MinDiff = st.number_input("Min Diff (Pos-Neg)", value=st.session_state.qual_params['MinDiff'])
            q_MinNeg = st.number_input("Min Neg", value=st.session_state.qual_params['MinNeg'])
            q_MaxNeg = st.number_input("Max Neg", value=st.session_state.qual_params['MaxNeg'])
            q_MinPos = st.number_input("Min Pos", value=st.session_state.qual_params['MinPos'])
            q_MaxPos = st.number_input("Max Pos", value=st.session_state.qual_params['MaxPos'])

        st.session_state.qual_params.update({
            'FNeg': q_FNeg, 'FPos': q_FPos, 'Const': q_Const,
            'MinDiff': q_MinDiff, 'MinNeg': q_MinNeg, 'MaxNeg': q_MaxNeg,
            'MinPos': q_MinPos, 'MaxPos': q_MaxPos
        })

# ==============================================================================
# MODE 1: ĐỊNH LƯỢNG (QUANTITATIVE)
# ==============================================================================
if app_mode == "1. Định lượng (Quantitative)":
    st.title("🧪 Mode 1: Định lượng (Recalibration)")
    st.markdown("**Nguyên lý:** Sử dụng 2 điểm Cal thực tế để nắn chỉnh đường Master Curve.")
    
    col_in, col_out = st.columns([1, 1.5])
    
    with col_in:
        st.subheader("1. Nhập kết quả Cal")
        # Nhập 2 điểm Cal
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("🔹 **Level 1**")
            t1 = st.number_input("Target 1:", value=42.1)
            s1 = st.number_input("Signal 1:", value=583722.0)
        with c2:
            st.markdown("🔹 **Level 2**")
            t2 = st.number_input("Target 2:", value=372.0)
            s2 = st.number_input("Signal 2:", value=288320.0)
            
        if st.button("🚀 Thực hiện Cal", type="primary"):
            p = st.session_state.master_params
            # Tính Master Signal
            ms1 = rod_4pl(t1, **p)
            ms2 = rod_4pl(t2, **p)
            
            # Tính Slope/Intercept
            if (ms2 - ms1) != 0:
                slope = (s2 - s1) / (ms2 - ms1)
                intercept = s1 - slope * ms1
                
                # Lưu vào Session State
                st.session_state.quant_results = {
                    'slope': slope, 'intercept': intercept,
                    't1': t1, 't2': t2, 's1': s1, 's2': s2,
                    'ms1': ms1, 'ms2': ms2
                }
                st.success("Đã Recalibration thành công!")
            else:
                st.error("Lỗi: Không thể tính toán (Target 1 và 2 giống nhau hoặc lỗi Master Curve).")

    with col_out:
        if st.session_state.quant_results:
            res = st.session_state.quant_results
            p = st.session_state.master_params
            
            st.subheader("2. Kết quả & Biểu đồ")
            
            # KPI
            k1, k2, k3 = st.columns(3)
            k1.metric("Slope (Factor)", f"{res['slope']:.4f}", help="Chuẩn: 0.8 - 1.2")
            k2.metric("Intercept", f"{res['intercept']:.0f}")
            status = "✅ PASS" if 0.8 <= res['slope'] <= 1.2 else "❌ FAIL"
            k3.metric("Trạng thái", status)
            
            # Biểu đồ
            x_plot = np.logspace(np.log10(5), np.log10(1000), 200)
            y_master = [rod_4pl(x, **p) for x in x_plot]
            y_recal = [y * res['slope'] + res['intercept'] for y in y_master]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_plot, y=y_master, mode='lines', name='Master (Gốc)', line=dict(dash='dash', color='gray')))
            fig.add_trace(go.Scatter(x=x_plot, y=y_recal, mode='lines', name='Hiện tại', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=[res['t1'], res['t2']], y=[res['s1'], res['s2']], mode='markers', name='Điểm Cal', marker=dict(size=12, color='red', symbol='cross')))
            fig.update_layout(xaxis_type="log", yaxis_type="log", height=400, xaxis_title="Nồng độ", yaxis_title="Tín hiệu")
            st.plotly_chart(fig, use_container_width=True)
            
            # Tính mẫu
            st.divider()
            st.subheader("3. Tính mẫu bệnh nhân")
            with st.form("quant_calc"):
                c_sig = st.number_input("Nhập Signal mẫu:", value=400000.0)
                if st.form_submit_button("Tính kết quả"):
                    norm_sig = (c_sig - res['intercept']) / res['slope']
                    final_conc = inv_rod_4pl(norm_sig, **p)
                    
                    st.success(f"Kết quả: **{final_conc:.4f}**")
                    # Vẽ điểm mẫu
                    fig.add_trace(go.Scatter(x=[final_conc], y=[c_sig], mode='markers', name='Mẫu', marker=dict(size=15, color='green', symbol='star')))
                    st.plotly_chart(fig, use_container_width=True, key='quant_chart_update')

# ==============================================================================
# MODE 2: ĐỊNH TÍNH (QUALITATIVE)
# ==============================================================================
elif app_mode == "2. Định tính (Qualitative)":
    st.title("⚖️ Mode 2: Định tính (Cutoff & COI)")
    st.markdown("**Nguyên lý:** Xác định điểm cắt (Cutoff) từ tín hiệu Âm/Dương tính.")
    
    qp = st.session_state.qual_params
    
    col_q_in, col_q_out = st.columns([1, 1.5])
    
    with col_q_in:
        st.subheader("1. Nhập tín hiệu Cal")
        sig_neg = st.number_input("Cal 1 (Negative):", value=2000.0)
        sig_pos = st.number_input("Cal 2 (Positive):", value=50000.0)
        
        if st.button("🚀 Xác lập Cutoff", type="primary"):
            # Kiểm tra QC
            msgs = []
            is_pass = True
            if not (qp['MinNeg'] <= sig_neg <= qp['MaxNeg']):
                msgs.append(f"❌ Neg ngoài dải ({qp['MinNeg']}-{qp['MaxNeg']})")
                is_pass = False
            if not (qp['MinPos'] <= sig_pos <= qp['MaxPos']):
                msgs.append(f"❌ Pos ngoài dải ({qp['MinPos']}-{qp['MaxPos']})")
                is_pass = False
            if (sig_pos - sig_neg) < qp['MinDiff']:
                msgs.append(f"❌ Khoảng cách Pos-Neg quá nhỏ (<{qp['MinDiff']})")
                is_pass = False
            
            # Tính Cutoff
            cutoff = (sig_neg * qp['FNeg']) + (sig_pos * qp['FPos']) + qp['Const']
            
            st.session_state.qual_results = {
                'cutoff': cutoff, 'is_pass': is_pass, 'msgs': msgs,
                'sig_neg': sig_neg, 'sig_pos': sig_pos
            }

    with col_q_out:
        if st.session_state.qual_results:
            res = st.session_state.qual_results
            
            st.subheader("2. Kết quả Calibration")
            if res['is_pass']:
                st.success(f"✅ PASSED | Cutoff = {res['cutoff']:,.0f}")
            else:
                st.error("⛔ FAILED")
                for m in res['msgs']: st.write(m)
            
            # Biểu đồ cột
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=['Neg', 'Cutoff', 'Pos'], y=[res['sig_neg'], res['cutoff'], res['sig_pos']], marker_color=['green', 'gray', 'red']))
            fig_bar.update_layout(title="Vị trí Cutoff", height=300)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            if res['is_pass']:
                st.divider()
                st.subheader("3. Tính COI (Index)")
                with st.form("qual_calc"):
                    s_sig = st.number_input("Signal mẫu:", value=100000.0)
                    if st.form_submit_button("Tính COI"):
                        coi = s_sig / res['cutoff']
                        concl = "DƯƠNG TÍNH" if coi >= 1.0 else "ÂM TÍNH"
                        color = "red" if coi >= 1.0 else "green"
                        
                        c1, c2 = st.columns(2)
                        c1.metric("COI", f"{coi:.2f}")
                        c2.markdown(f"### :{color}[{concl}]")
                        
                        # Vẽ điểm
                        fig_bar.add_trace(go.Scatter(x=['Mẫu'], y=[s_sig], mode='markers', marker=dict(size=15, color='orange', symbol='star')))
                        st.plotly_chart(fig_bar, use_container_width=True, key='qual_chart_upd')

# ==============================================================================
# MODE 3: TROUBLESHOOT (LỊCH SỬ)
# ==============================================================================
elif app_mode == "3. Troubleshoot (Lịch sử)":
    st.title("📈 Mode 3: Trend Analysis (Phân tích Xu hướng)")
    st.markdown("Nhập dữ liệu lịch sử để kiểm tra độ ổn định của hệ thống.")
    
    # Dữ liệu mẫu
    df_sample = pd.DataFrame([
        {"Date": "2023-12-01", "Target 1": 42.1, "Target 2": 372.0, "Signal 1": 590000, "Signal 2": 295000},
        {"Date": "2023-12-15", "Target 1": 42.1, "Target 2": 372.0, "Signal 1": 583602, "Signal 2": 289073},
        {"Date": "2023-12-25", "Target 1": 42.1, "Target 2": 372.0, "Signal 1": 550000, "Signal 2": 270000},
    ])
    
    st.subheader("1. Nhập lịch sử Cal")
    edited_df = st.data_editor(df_sample, num_rows="dynamic", use_container_width=True)
    
    if st.button("🔍 Phân tích", type="primary"):
        p = st.session_state.master_params
        results = []
        
        for i, row in edited_df.iterrows():
            try:
                t1, t2 = float(row['Target 1']), float(row['Target 2'])
                s1, s2 = float(row['Signal 1']), float(row['Signal 2'])
                
                m1 = rod_4pl(t1, **p)
                m2 = rod_4pl(t2, **p)
                
                slope = (s2 - s1) / (m2 - m1)
                results.append({"Date": row['Date'], "Slope": slope, "S1": s1, "S2": s2, "T1": t1, "T2": t2})
            except: pass
            
        res_df = pd.DataFrame(results)
        
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Xu hướng Slope")
            fig_tr = go.Figure()
            fig_tr.add_hrect(y0=0.8, y1=1.2, fillcolor="green", opacity=0.1, line_width=0)
            fig_tr.add_trace(go.Scatter(x=res_df['Date'], y=res_df['Slope'], mode='lines+markers', name='Slope'))
            fig_tr.update_layout(yaxis_title="Slope Factor", height=400)
            st.plotly_chart(fig_tr, use_container_width=True)
            
        with c2:
            st.subheader("Độ lệch Master Curve")
            x_d = np.logspace(np.log10(5), np.log10(1000), 200)
            y_m = [rod_4pl(x, **p) for x in x_d]
            
            fig_ov = go.Figure()
            fig_ov.add_trace(go.Scatter(x=x_d, y=y_m, mode='lines', name='Master', line=dict(color='gray')))
            for i, r in res_df.iterrows():
                fig_ov.add_trace(go.Scatter(x=[r['T1'], r['T2']], y=[r['S1'], r['S2']], mode='lines+markers', name=str(r['Date']), opacity=0.5))
            fig_ov.update_layout(xaxis_type="log", yaxis_type="log", height=400)
            st.plotly_chart(fig_ov, use_container_width=True)
