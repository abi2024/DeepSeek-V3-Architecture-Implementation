import streamlit as st
import json
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide", page_title="MoE Microscope")

st.title("🔬 MoE Microscope: DeepSeek Monitor")

LOG_FILE = "logs/training_metrics.json"

def load_data():
    if not os.path.exists(LOG_FILE):
        return None
    try:
        with open(LOG_FILE, 'r') as f:
            data = json.load(f)
        return data
    except:
        return None

if st.button('🔄 Refresh Data'):
    st.rerun()

data = load_data()

if not data:
    st.warning("⚠️ No logs found yet. Training needs to reach Step 1000 (or first checkpoint) to generate logs.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(data)

# --- 1. Global Metrics ---
st.header("1. Training Health")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Router Entropy (Diversity)")
    layers_to_plot = [col for col in df.columns if 'entropy' in col]
    if layers_to_plot:
        # Select Start, Middle, End layers dynamically
        mid = len(layers_to_plot) // 2
        selected_layers = [layers_to_plot[0], layers_to_plot[mid], layers_to_plot[-1]]
        # Deduplicate in case fewer than 3 layers
        selected_layers = list(set(selected_layers))
        
        fig_ent = px.line(df, x='step', y=selected_layers, title="Router Entropy (Higher = More Random)")
        st.plotly_chart(fig_ent, use_container_width=True)

with col2:
    st.subheader("Token Drop Rate")
    drop_cols = [col for col in df.columns if 'droprate' in col]
    if drop_cols:
        df['avg_drop_rate'] = df[drop_cols].mean(axis=1)
        fig_drop = px.area(df, x='step', y='avg_drop_rate', title="Average Token Drop Rate")
        st.plotly_chart(fig_drop, use_container_width=True)

# --- 2. Expert Load Heatmap ---
st.header("2. Expert Load Heatmap")
st.markdown("Visualizing which experts are being used. **Vertical lines = Good.** **Horizontal lines = Collapse.**")

if not df.empty:
    # Selector for specific step
    steps = df['step'].tolist()
    selected_step = st.select_slider("Select Training Step", options=steps, value=steps[-1])
    step_data = df[df['step'] == selected_step].iloc[0]

    # Build Heatmap Data (Layers x Experts)
    load_data = []
    load_cols = [c for c in df.columns if 'load' in c]
    
    # Robust Sort: Extract integer layer index 'layer_10_load' -> 10
    def get_layer_idx(col_name):
        try:
            return int(col_name.split('_')[1])
        except:
            return 0
            
    load_cols.sort(key=get_layer_idx)

    for col in load_cols:
        load_data.append(step_data[col])

    # [Layers, Experts]
    heatmap_matrix = load_data 
    
    # DYNAMIC DIMENSIONS FIX
    num_layers = len(heatmap_matrix)
    num_experts = len(heatmap_matrix[0]) if num_layers > 0 else 8

    fig_heat = px.imshow(
        heatmap_matrix,
        labels=dict(x="Expert ID", y="Layer ID", color="Load %"),
        x=[f"Exp {i}" for i in range(num_experts)],
        y=[f"L{i}" for i in range(num_layers)], # FIXED: Uses actual data length
        aspect="auto",
        title=f"Expert Utilization at Step {selected_step}"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.caption(f"Data Source: {LOG_FILE}")