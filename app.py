import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import json
from openai import OpenAI
# Streamlit components for HTML/JS rendering
import streamlit.components.v1 as components

# ==========================================
# 1. Page Configuration & Title
# ==========================================
st.set_page_config(page_title="COF Hydrogen Storage Evaluation System", page_icon="⚛️", layout="wide")

st.title("⚛️ COF Hydrogen Storage Evaluation System (XAI-LLM Driven)")
st.markdown("""
This system is based on **Explainable AI (PDP rules)** and **Large Language Models (LLM)**.
3D structures are rendered directly via HTML/JS technology, requiring no third-party Python plugins.
""")
st.divider()

# ==========================================
# 2. Sidebar: Input Parameters
# ==========================================
# --- University Branding (Xi'an Jiaotong University) ---
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-top: -20px; margin-bottom: 20px;">
        <h3 style="color: #003366; font-family: 'Times New Roman', Times, serif; font-weight: bold;">
            Xi'an Jiaotong University
        </h3>
        <hr style="border-top: 2px solid #003366; margin-top: -10px;">
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("📥 Input COF Parameters")

cof_name = st.sidebar.text_input("Material Name (Must match file in structures/)", value="Test-COF-01")
density = st.sidebar.number_input("Density (g/cm³)", value=0.45, min_value=0.01, max_value=3.0, step=0.05)
sa = st.sidebar.number_input("Surface Area SA (m²/g)", value=2500.0, min_value=0.0, max_value=8000.0, step=100.0)
vf = st.sidebar.number_input("Pore Volume Vf (cm³/g)", value=1.5, min_value=0.0, max_value=6.0, step=0.1)
porosity = st.sidebar.slider("Porosity (φ)", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

st.sidebar.divider()
st.sidebar.header("⚙️ LLM Engine Settings")
api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password", help="If empty, local XAI logic will be used.")

# ==========================================
# 3. Core Functions
# ==========================================

# 3D Rendering (Native Component approach)
def render_3d_mol(cif_text):
    mol_html = f"""
    <div id="container-3d" style="height: 400px; width: 100%; position: relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        var element = document.getElementById('container-3d');
        var config = {{ backgroundColor: 'white' }};
        var viewer = $3Dmol.createViewer(element, config);
        viewer.addModel(`{cif_text}`, "cif");
        viewer.setStyle({{stick: {{colorscheme: 'Jmol', radius: 0.2}}}});
        viewer.zoomTo();
        viewer.render();
        viewer.spin(true); 
    </script>
    """
    components.html(mol_html, height=420)

# LLM Evaluation Function
def get_evaluation(density, sa, vf, porosity, api_key):
    system_prompt = f"""
    You are a materials science expert. Evaluate COF hydrogen storage grade (S/A/B/C) based on:
    Density: {density}, SA: {sa}, Vf: {vf}, Porosity: {porosity}
    Return ONLY a JSON: {{"Grade": "...", "Predicted_Capacity": "...", "Diagnostic_Report": "..."}}
    (Use English for the report)
    """
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": system_prompt}],
                temperature=0.2
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"Grade": "Error", "Predicted_Capacity": "N/A", "Diagnostic_Report": f"API Error: {e}"}
    else:
        # Mock Local Logic
        time.sleep(1)
        if sa > 3500: 
            return {"Grade": "S", "Predicted_Capacity": "> 6.5 wt%", "Diagnostic_Report": "[Local XAI] This material exhibits ultra-high surface area, promising excellent adsorption performance."}
        return {"Grade": "B", "Predicted_Capacity": "3.0-5.5 wt%", "Diagnostic_Report": "[Local XAI] Moderate structural parameters. Optimization of pore geometry is recommended."}

# ==========================================
# 4. Main Layout (Three Columns)
# ==========================================

current_dir = os.path.dirname(os.path.abspath(__file__))
cif_path = os.path.join(current_dir, "structures", f"{cof_name}.cif")
has_cif = os.path.exists(cif_path)

if has_cif:
    col1, col2, col3 = st.columns([1, 1, 1.2])
else:
    col1, col2 = st.columns([1, 2])

# --- Column 1: Radar Chart ---
with col1:
    st.subheader("📊 Structural Radar")
    categories = ['Surface Area', 'Pore Volume', 'Porosity', 'Lightweight']
    ideal_values = [1.0, 1.0, 0.9, 1.0]
    input_values = [min(sa/4000, 1.0), min(vf/2.5, 1.0), porosity, min(0.5/density, 1.0)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ideal_values, theta=categories, fill='toself', name='Target'))
    fig.add_trace(go.Scatterpolar(r=input_values, theta=categories, fill='toself', name=cof_name))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

# --- Column 2: LLM Diagnosis ---
with col2:
    st.subheader("🧠 AI Diagnostic Results")
    if st.button("🚀 Run Assessment", type="primary"):
        with st.spinner('Analyzing physics and chemical properties...'):
            result = get_evaluation(density, sa, vf, porosity, api_key)
            st.metric("Final Grade", f"Grade {result.get('Grade', result.get('等级'))}")
            st.write(f"**Estimated Capacity:** {result.get('Predicted_Capacity', result.get('预测吸附量范围'))}")
            st.info(f"**Expert Report:** {result.get('Diagnostic_Report', result.get('诊断报告'))}")
    else:
        st.write("Click the button to start the AI diagnostic.")

# --- Column 3: 3D Visualization ---
if has_cif:
    with col3:
        st.subheader("🧊 3D Structure Preview")
        try:
            with open(cif_path, "r") as f:
                cif_content = f.read()
            render_3d_mol(cif_content)
            st.caption(f"Currently displaying: {cof_name}.cif (Rotate/Zoom with mouse)")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ==========================================
# 5. Footer
# ==========================================
st.divider()
st.caption("Powered by LLM & XAI | 3D Visualization via 3Dmol.js CDN")
