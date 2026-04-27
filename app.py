import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
from openai import OpenAI

# ==========================================
# 1. 页面配置与标题
# ==========================================
st.set_page_config(page_title="COF 储氢潜力在线评估系统", page_icon="⚛️", layout="wide")

st.title("⚛️ COF 储氢潜力评估系统 (XAI-LLM Driven)")
st.markdown("""
参考 DOE (美国能源部) 储氢目标，本系统基于 **可解释机器学习 (PDP法则)** 与 **大语言模型 (LLM)** 协同计算。
只需输入目标共价有机框架 (COF) 的几何特征参数，系统将自动进行物理规律判定，并输出 DOE 评级与诊断报告。
""")
st.divider()

# ==========================================
# 2. 侧边栏：输入材料特征参数
# ==========================================
st.sidebar.header("📥 输入 COF 材料参数")
st.sidebar.markdown("请根据 GCMC 模拟或实验数据输入：")

cof_name = st.sidebar.text_input("材料名称 (如 ZnP-COF)", value="Test-COF-01")
density = st.sidebar.number_input("骨架密度 Density (g/cm³)", value=0.45, min_value=0.01, max_value=3.0, step=0.05)
sa = st.sidebar.number_input("质量比表面积 SA (m²/g)", value=2500.0, min_value=0.0, max_value=8000.0, step=100.0)
vf = st.sidebar.number_input("孔体积 Vf (cm³/g)", value=1.5, min_value=0.0, max_value=6.0, step=0.1)
porosity = st.sidebar.slider("孔隙率 Porosity (φ)", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

# API Key 设置 (让用户在侧边栏输入他们的 API Key)
st.sidebar.divider()
st.sidebar.header("⚙️ LLM 引擎设置")
api_key = st.sidebar.text_input("OpenAI API Key (可选)", type="password", help="如果为空，将使用内置的 XAI 规则进行模拟演示")

# ==========================================
# 3. 核心功能：提示词组装与大模型调用
# ==========================================
def get_evaluation(density, sa, vf, porosity, api_key):
    # 组装 XAI 提取的物理法则 Prompt
    system_prompt = f"""
    你是一个权威的材料科学与新能源专家。请根据以下 COF 材料参数，评估其是否能达到美国能源部（DOE）的储氢目标。
    【当前材料参数】：
    - 密度: {density} g/cm³
    - 质量比表面积: {sa} m²/g
    - 孔体积: {vf} cm³/g
    - 孔隙率: {porosity}

    【DOE 评估等级及 XAI 判定法则】：
    - **S级 (超额完成, >6.5 wt%)**: 质量比表面积 > 3500 且 孔体积 > 2.0 且 密度 < 0.5。
    - **A级 (符合近期目标, 5.5~6.5 wt%)**: 质量比表面积 2500~3500，孔体积 > 1.5。
    - **C级 (极其劣质, <3.0 wt%)**: 【一票否决】如果密度 > 1.2 g/cm³ 或 比表面积 < 500。
    - **B级 (潜力一般, 3.0~5.5 wt%)**: 介于 A 与 C 之间。

    请严格按以下 JSON 格式返回结果（不要输出任何其他多余文本）：
    {{"等级": "S/A/B/C 中的一个", "预测吸附量范围": "例如 5.5-6.5 wt%", "诊断报告": "一段详细的物理化学诊断分析（150字左右），解释为什么是这个等级"}}
    """

    # 如果有 API Key，则调用真实大模型
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # 也可以换成 gpt-4o
                messages=[{"role": "user", "content": system_prompt}],
                temperature=0.2 # 低温度保证逻辑严谨
            )
            # 这是一个简化的解析，实际使用中建议用 JSON mode
            import json
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"等级": "Error", "预测吸附量范围": "N/A", "诊断报告": f"API调用失败: {e}"}
    
    # 如果没有 API Key，使用内置的 If-Else 规则模拟大模型输出（方便你本地测试）
    else:
        time.sleep(1.5) # 模拟思考时间
        if density > 1.2 or sa < 500:
            return {"等级": "C", "预测吸附量范围": "< 3.0 wt%", "诊断报告": "【系统模拟输出】该材料触发了一票否决机制。密度过高（>1.2）导致结构过度致密，或者比表面积过小，无法为氢气分子提供有效的物理吸附位点，不具备工业应用潜力。"}
        elif sa > 3500 and vf > 2.0 and density < 0.5:
            return {"等级": "S", "预测吸附量范围": "> 6.5 wt%", "诊断报告": "【系统模拟输出】完美！该材料拥有极其优异的空腔结构和超高的孔隙率，质量比表面积突破 3500 m²/g 阈值。推测其具有极强的多层物理吸附能力，有望超越 DOE 终极目标。"}
        elif sa >= 2500 and vf >= 1.5:
            return {"等级": "A", "预测吸附量范围": "5.5 ~ 6.5 wt%", "诊断报告": "【系统模拟输出】表现优秀。该材料具备充足的吸附位点（比表面积达标）和足够的孔体积容量，能够满足 DOE 2025 年近期质量储氢目标，是值得实验合成的优质候选材料。"}
        else:
            return {"等级": "B", "预测吸附量范围": "3.0 ~ 5.5 wt%", "诊断报告": "【系统模拟输出】潜力中等。虽然具备一定的多孔结构，但比表面积和孔隙容量尚未达到高阶协同效应的阈值，建议通过掺杂金属或改变连接基团来优化孔径结构。"}

# ==========================================
# 4. 主界面：可视化与结果展示
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 结构参数雷达图")
    # 绘制雷达图对比理想目标
    categories = ['比表面积 (归一化)', '孔体积 (归一化)', '孔隙率', '轻量化程度 (1/密度)']
    
    # 假设理想目标 (S级) 的参数基准
    ideal_values = [1.0, 1.0, 0.9, 1.0] 
    
    # 当前输入归一化计算 (仅为作图展示)
    input_values = [min(sa/4000, 1.0), min(vf/2.5, 1.0), porosity, min(0.5/density, 1.0)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ideal_values, theta=categories, fill='toself', name='DOE 终极目标', line=dict(color='rgba(0, 255, 0, 0.5)')))
    fig.add_trace(go.Scatterpolar(r=input_values, theta=categories, fill='toself', name=cof_name, line=dict(color='blue')))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1.2])), showlegend=True, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🧠 大模型诊断结果")
    if st.button("🚀 启动 LLM 评估", type="primary"):
        with st.spinner('正在组装 XAI 提示词... 正在调用大语言模型进行物理推理...'):
            result = get_evaluation(density, sa, vf, porosity, api_key)
            
            if result["等级"] == "S":
                st.success(f"### 🏆 最终评级：{result['等级']} 级 (超额达标)")
            elif result["等级"] == "A":
                st.info(f"### 🏅 最终评级：{result['等级']} 级 (符合近期目标)")
            elif result["等级"] == "B":
                st.warning(f"### ⚠️ 最终评级：{result['等级']} 级 (需要进一步修饰)")
            else:
                st.error(f"### ❌ 最终评级：{result['等级']} 级 (建议淘汰)")
            
            st.markdown(f"**📈 预测吸附量：** `{result['预测吸附量范围']}`")
            
            st.markdown("#### 🔬 AI 专家诊断报告")
            st.info(result["诊断报告"])
    else:
        st.write("👈 请在左侧输入材料参数，并点击上方按钮进行智能诊断。")

# 页脚
st.divider()
st.caption("Powered by LLM & Explainable AI (PDP) | Open-source material screening framework.")
