import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import json
from openai import OpenAI
# 引入 Streamlit 原生组件，用于渲染 HTML/JavaScript
import streamlit.components.v1 as components

# ==========================================
# 1. 页面配置与标题
# ==========================================
st.set_page_config(page_title="COF 储氢潜力在线评估系统", page_icon="⚛️", layout="wide")

st.title("⚛️ COF 储氢潜力评估系统 (XAI-LLM Driven)")
st.markdown("""
本系统基于 **可解释机器学习 (PDP法则)** 与 **大语言模型 (LLM)**。
通过 HTML/JS 技术直接渲染 3D 结构，无需第三方 Python 插件。
""")
st.divider()

# ==========================================
# 2. 侧边栏：输入参数
# ==========================================
# --- 新增：学校标识 (Xi'an Jiaotong University) ---
# 使用 Markdown 在侧边栏最顶部插入学校名字
# 这里使用了居中和深蓝色 (#003366)，使其看起来更正式
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
# --------------------------------------------------
# ... 原有的输入参数代码保持不变 ...
st.sidebar.header("📥 输入 COF 材料参数")

cof_name = st.sidebar.text_input("材料名称 (需对应 structures/ 目录下的文件名)", value="Test-COF-01")
density = st.sidebar.number_input("骨架密度 Density (g/cm³)", value=0.45, min_value=0.01, max_value=3.0, step=0.05)
sa = st.sidebar.number_input("质量比表面积 SA (m²/g)", value=2500.0, min_value=0.0, max_value=8000.0, step=100.0)
vf = st.sidebar.number_input("孔体积 Vf (cm³/g)", value=1.5, min_value=0.0, max_value=6.0, step=0.1)
porosity = st.sidebar.slider("孔隙率 Porosity (φ)", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

st.sidebar.divider()
st.sidebar.header("⚙️ LLM 引擎设置")
api_key = st.sidebar.text_input("OpenAI API Key (可选)", type="password")

# ==========================================
# 3. 核心功能函数
# ==========================================

# 3D 渲染函数 (已修改为原生组件方案，解决部署报错)
def render_3d_mol(cif_text):
    # 构建嵌入式 HTML 代码，直接从 3Dmol.org 的 CDN 加载脚本
    mol_html = f"""
    <div id="container-3d" style="height: 400px; width: 100%; position: relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        var element = document.getElementById('container-3d');
        var config = {{ backgroundColor: 'white' }};
        var viewer = $3Dmol.createViewer(element, config);
        // 使用反引号处理可能包含换行符的 CIF 文本
        viewer.addModel(`{cif_text}`, "cif");
        viewer.setStyle({{stick: {{colorscheme: 'Jmol', radius: 0.2}}}});
        viewer.zoomTo();
        viewer.render();
        viewer.spin(true); // 开启自动旋转
    </script>
    """
    # 渲染 HTML 组件
    components.html(mol_html, height=420)

# LLM 评估函数
def get_evaluation(density, sa, vf, porosity, api_key):
    system_prompt = f"""
    你是一个材料科学专家。根据以下参数评估 COF 储氢等级 (S/A/B/C)：
    密度: {density}, 比表面积: {sa}, 孔体积: {vf}, 孔隙率: {porosity}
    请按 JSON 格式返回：{{"等级": "...", "预测吸附量范围": "...", "诊断报告": "..."}}
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
            return {"等级": "Error", "预测吸附量范围": "N/A", "诊断报告": f"API失败: {e}"}
    else:
        # 模拟内置逻辑
        time.sleep(1)
        if sa > 3500: return {"等级": "S", "预测吸附量范围": "> 6.5 wt%", "诊断报告": "该材料具有极高比表面积，表现优异。"}
        return {"等级": "B", "预测吸附量范围": "3.0-5.5 wt%", "诊断报告": "性能中等，建议优化。"}

# ==========================================
# 4. 主界面布局 (调整为三列)
# ==========================================

# 显式路径处理，确保在云端准确寻找文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
cif_path = os.path.join(current_dir, "structures", f"{cof_name}.cif")
has_cif = os.path.exists(cif_path)

# 根据是否有 CIF 文件决定列的分配
if has_cif:
    col1, col2, col3 = st.columns([1, 1, 1.2])
else:
    col1, col2 = st.columns([1, 2])

# --- 第一列：雷达图 ---
with col1:
    st.subheader("📊 结构参数雷达图")
    categories = ['比表面积', '孔体积', '孔隙率', '轻量化']
    ideal_values = [1.0, 1.0, 0.9, 1.0]
    input_values = [min(sa/4000, 1.0), min(vf/2.5, 1.0), porosity, min(0.5/density, 1.0)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ideal_values, theta=categories, fill='toself', name='目标'))
    fig.add_trace(go.Scatterpolar(r=input_values, theta=categories, fill='toself', name=cof_name))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

# --- 第二列：LLM 诊断 ---
with col2:
    st.subheader("🧠 大模型诊断结果")
    if st.button("🚀 启动评估", type="primary"):
        with st.spinner('计算中...'):
            result = get_evaluation(density, sa, vf, porosity, api_key)
            st.metric("最终评级", f"{result['等级']} 级")
            st.write(f"**吸附量范围:** {result['预测吸附量范围']}")
            st.info(f"**诊断报告:** {result['诊断报告']}")
    else:
        st.write("点击按钮开始评估")

# --- 第三列：3D 可视化 ---
if has_cif:
    with col3:
        st.subheader("🧊 3D 结构预览")
        try:
            with open(cif_path, "r") as f:
                cif_content = f.read()
            # 渲染 3D 结构
            render_3d_mol(cif_content)
            st.caption(f"当前显示: {cof_name}.cif (可通过鼠标旋转/缩放)")
        except Exception as e:
            st.error(f"无法读取文件: {e}")

# ==========================================
# 5. 页脚
# ==========================================
st.divider()
st.caption("Powered by LLM & XAI | 3D Visualization via 3Dmol.js CDN")
