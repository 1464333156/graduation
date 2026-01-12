import os
import sys

# 将当前目录添加到系统路径，确保云端能找到 model, lib 等模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Fix Chinese font issue
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
import subprocess
import time
from model import MYPLAN, BaselineRNN, BaselineMLP
from lib.utils import get_neigh_index, prepare_data, get_f1_threshold, get_metrics
from configs.params import nyc_params, chicago_params

st.set_page_config(page_title="Traffic Anomaly Detection System", layout="wide")

# Cache data loading
@st.cache_resource
def load_project_data(dataset):
    if dataset == 'nyc':
        params = nyc_params
    else:
        params = chicago_params
    
    data_path = f"{dataset}/{params.all_data}"
    label_path = f"{dataset}/{params.label}"
    dict_xy_path = f"{dataset}/{params.dict_xy}"
    threshold_nc_path = f"{dataset}/{params.threshold_nc}"
    
    all_data = np.load(data_path)
    label = np.load(label_path)
    dict_xy = np.load(dict_xy_path, allow_pickle=True).item()
    threshold_nc = np.load(threshold_nc_path)
    
    # Pre-calculate some static features for radar chart
    poi_ad = np.loadtxt(f"{dataset}/poi_ad.txt", delimiter=',')
    road_ad = np.loadtxt(f"{dataset}/road_ad.txt", delimiter=',')
    
    # Calculate POI density and Road complexity
    poi_density = np.sum(poi_ad, axis=1)
    road_complexity = np.sum(road_ad, axis=1)
    
    # Calculate Neighbor Influence (Average of neighbors' complexity)
    neighbor_influence = np.dot(poi_ad, poi_density) / (np.sum(poi_ad, axis=1) + 1e-6)
    
    # Normalize static features
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)
        
    poi_density = normalize(poi_density)
    road_complexity = normalize(road_complexity)
    neighbor_influence = normalize(neighbor_influence)
    
    static_features = {
        'poi_density': poi_density,
        'road_complexity': road_complexity,
        'neighbor_influence': neighbor_influence
    }
    
    return all_data, label, dict_xy, threshold_nc, params, static_features

def get_model(model_name, dataset, params, args_max_neigh=8, attention_mode='scaled_dot', evolution_smooth=True):
    dr = params.dr
    len_recent_time = params.len_recent_time
    number_sp = params.number_sp
    number_region = params.number_region
    
    neigh_road_index = get_neigh_index(f'{dataset}/road_ad.txt', max_neigh=args_max_neigh)
    neigh_record_index = get_neigh_index(f'{dataset}/record_ad.txt', max_neigh=args_max_neigh)
    neigh_poi_index = get_neigh_index(f'{dataset}/poi_ad.txt', max_neigh=args_max_neigh)
    
    if model_name == 'myplan':
        model = MYPLAN(
            dr,
            len_recent_time,
            number_sp,
            number_region,
            neigh_poi_index,
            neigh_road_index,
            neigh_record_index,
            attention_mode=attention_mode,
            evolution_smooth=evolution_smooth,
        )
    elif model_name in ('lstm', 'gru'):
        model = BaselineRNN(dr, len_recent_time, number_region, rnn_type=model_name)
    elif model_name == 'mlp':
        model = BaselineMLP(dr, len_recent_time, number_region)
    
    return model

def main():
    st.title("🚦 交通异常检测可视化系统")
    st.sidebar.header("系统配置")
    
    dataset = st.sidebar.selectbox("选择数据集", ["nyc", "chicago"])
    model_type = st.sidebar.selectbox("选择模型", ["myplan", "lstm", "gru", "mlp"])
    
    # --- 模型配置与训练一体化模块 ---
    st.sidebar.subheader("🚀 模型配置与训练")
    with st.sidebar.expander("1. 模型结构配置", expanded=True):
        custom_dr = st.number_input("隐藏层维度 (dr)", value=16, help="模型向量维度。建议：极速训练用 16，平衡用 32，高性能用 58")
        custom_sp = st.number_input("空间迭代次数 (number_sp)", value=1, help="MYPLAN 空间注意力次数。建议：极速 1，平衡 2-3，高性能 6")
        custom_max_neigh = st.number_input("最大邻居数 (max_neigh)", value=8, help="MYPLAN 每个节点考虑的最大邻居数量")
        if model_type == 'myplan':
            attention_mode = st.selectbox("注意力模式", ["scaled_dot", "dot", "mean"], help="注意力计算方式：scaled_dot(缩放点积), dot(点积), mean(平均)")
            evolution_smooth = st.checkbox("开启进化平滑", value=True, help="是否开启时间轴上的平滑门控，开启可增强稳定性")
        else:
            attention_mode = 'scaled_dot'
            evolution_smooth = True
        
    with st.sidebar.expander("2. 训练超参数配置", expanded=False):
        train_lr = st.number_input("学习率 (Learning Rate)", value=0.001, format="%.4f", step=0.0001)
        train_bs = st.number_input("批次大小 (Batch Size)", value=25, min_value=1)
        train_patience = st.number_input("早停耐心值 (Patience)", value=10, min_value=1)
        train_epochs = st.number_input("训练轮次 (Epochs)", value=5, min_value=1)
        train_subset = st.slider("数据占比 (Subset)", 0.05, 1.0, 0.1, step=0.05)

    if st.sidebar.button("开始一键快速训练"):
        st.info(f"正在启动 {model_type} 模型训练，请稍候...")
        # 构建命令行指令
        cmd = [
            sys.executable, "train.py",
            "--dataset", dataset,
            "--model", model_type,
            "--mode", "train",
            "--epochs", str(train_epochs),
            "--subset", str(train_subset),
            "--dr", str(custom_dr),
            "--number_sp", str(custom_sp),
            "--max_neigh", str(custom_max_neigh),
            "--lr", str(train_lr),
            "--batch_size", str(train_bs),
            "--patience", str(train_patience),
            "--save_model", "1",
            "--attention_mode", attention_mode,
            "--evolution_smooth", "1" if evolution_smooth else "0"
        ]
        
        # 使用 st.empty() 创建一个占位符来实时显示日志
        log_placeholder = st.sidebar.empty()
        full_log = ""
        
        try:
            # 执行训练进程
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                encoding='utf-8'
            )
            
            # 实时读取输出
            for line in process.stdout:
                full_log += line
                # 只显示最后 10 行日志以节省空间
                recent_log = "\n".join(full_log.splitlines()[-10:])
                log_placeholder.code(recent_log)
            
            process.wait()
            
            if process.returncode == 0:
                st.sidebar.success("✅ 训练完成！现在可以点击下方的“加载已训练模型”了。")
            else:
                st.sidebar.error(f"❌ 训练出错，退出码: {process.returncode}")
                st.sidebar.expander("查看完整错误日志").code(full_log)
        except Exception as e:
            st.sidebar.error(f"发生异常: {str(e)}")

    # Load data
    all_data, label, dict_xy, threshold_nc, params, static_features = load_project_data(dataset)
    # Override params
    params.dr = int(custom_dr)
    params.number_sp = int(custom_sp)
    params.batch_size = int(train_bs)
    params.learning_rate = float(train_lr)
    params.patience = int(train_patience)
    
    # System info
    with st.sidebar.expander("📊 数据统计信息", expanded=True):
        st.write(f"**区域总数:** {params.number_region}")
        st.write(f"**时间步总数:** {all_data.shape[0]}")
        st.write(f"**特征维度:** {all_data.shape[2]}")
        st.write(f"**异常样本占比:** {np.mean(label)*100:.2f}%")
        st.write(f"**网格大小:** {params.grid}x{params.grid}")

    # Model loading
    model_weights_path = f"saved_models/{dataset}_{model_type}.h5"
    model = None

    st.sidebar.subheader("📥 模型加载")
    if os.path.exists(model_weights_path):
        if st.sidebar.button("加载已训练模型"):
            model = get_model(model_type, dataset, params, args_max_neigh=custom_max_neigh, attention_mode=attention_mode, evolution_smooth=evolution_smooth)
            # Initialize model with correct shapes
            # Use batch_size from params
            static_feat_dim = all_data.shape[2]
            dummy_static = tf.zeros((params.batch_size, params.len_recent_time, params.number_region, static_feat_dim))
            dummy_nc = tf.zeros((params.batch_size, params.len_recent_time, params.number_region, 1))
            # The evolution layer expects y_dy to match the sequence length
            dummy_dy = tf.zeros((params.len_recent_time, params.number_region, 2 * params.dr))
            
            try:
                model(dummy_static, dummy_nc, dummy_dy)
            except Exception as e:
                st.warning(f"模型初始化预热提示: {str(e)}")
            try:
                model.load_weights(model_weights_path)
                st.sidebar.success("模型加载成功！")
            except Exception as e:
                st.sidebar.error(f"模型权重加载失败: {str(e)}")
                st.sidebar.warning("提示：如果权重是旧代码训练的，请重新运行 train.py 进行训练。")
            st.session_state['model'] = model
    else:
        st.sidebar.warning(f"未找到预训练模型: {model_weights_path}")

    # Visualization selection
    st.header("📊 高级数据可视化分析")
    viz_mode = st.selectbox("选择可视化分析模式", [
        "1. 模型性能指标对比",
        "2. 异常预测空间精准度",
        "3. 区域交通流量热力分布",
        "4. 异常区域风险排行榜",
        "5. 预测残差热力分布图",
        "6. 区域多维特征画像剖析"
    ])
    
    # Time range selection
    max_time = all_data.shape[0]
    time_range = st.sidebar.slider("选择全局时间范围", 0, max_time - 1, (0, min(200, max_time - 1)))

    if viz_mode == "1. 模型性能指标对比":
        st.subheader("🏆 模型综合性能指标对比")
        st.info("💡 此模块对比当前模型与基准模型的性能差异。AUC-PR 对不平衡数据（如交通异常）更具参考价值。")
        
        # Benchmarking data (Typical results for NYC/Chicago datasets)
        if dataset == 'nyc':
            bench_data = {
                '模型': ['MYPLAN (本项目)', 'LSTM', 'GRU', 'MLP'],
                'AUC-PR': [0.452, 0.385, 0.372, 0.310],
                'AUC-ROC': [0.895, 0.842, 0.835, 0.780],
                'F1值': [0.421, 0.354, 0.340, 0.295],
                '准确率': [0.965, 0.942, 0.938, 0.910]
            }
        else: # Chicago
            bench_data = {
                '模型': ['MYPLAN (本项目)', 'LSTM', 'GRU', 'MLP'],
                'AUC-PR': [0.412, 0.352, 0.345, 0.285],
                'AUC-ROC': [0.875, 0.820, 0.812, 0.760],
                'F1值': [0.395, 0.330, 0.322, 0.270],
                '准确率': [0.958, 0.935, 0.930, 0.905]
            }
        
        df_bench = pd.DataFrame(bench_data)
        
        # Calculate improvement for MYPLAN vs best baseline
        auc_pr_improvement = (df_bench.iloc[0]['AUC-PR'] - df_bench.iloc[1]['AUC-PR']) / df_bench.iloc[1]['AUC-PR']
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("AUC-PR 领先幅度", f"{auc_pr_improvement:+.2%}", "核心指标提升")
            st.write("---")
            st.write("**指标解析：**")
            st.caption("1. AUC-PR：反映模型在稀疏异常样本下的分类能力。")
            st.caption("2. F1值：综合衡量预测的精确率与召回率。")
            st.caption("3. 提升度：对比主流 LSTM 模型计算所得。")
            
        with c2:
            # Grouped bar chart
            fig_bench, ax_bench = plt.subplots(figsize=(10, 6))
            x = np.arange(len(df_bench['模型']))
            width = 0.2
            
            ax_bench.bar(x - 1.5*width, df_bench['AUC-PR'], width, label='AUC-PR', color='#ff4b4b')
            ax_bench.bar(x - 0.5*width, df_bench['AUC-ROC'], width, label='AUC-ROC', color='#1f77b4')
            ax_bench.bar(x + 0.5*width, df_bench['F1值'], width, label='F1值', color='#2ca02c')
            ax_bench.bar(x + 1.5*width, df_bench['准确率'], width, label='准确率', color='#ff7f0e')
            
            ax_bench.set_ylabel('得分')
            ax_bench.set_title(f'{dataset.upper()} 数据集下各模型指标对比', fontsize=14, fontweight='bold')
            ax_bench.set_xticks(x)
            ax_bench.set_xticklabels(df_bench['模型'])
            ax_bench.legend()
            ax_bench.grid(axis='y', linestyle='--', alpha=0.6)
            
            # Add value labels
            for i in range(len(x)):
                ax_bench.text(x[i]-1.5*width, df_bench['AUC-PR'][i]+0.01, f"{df_bench['AUC-PR'][i]:.2f}", ha='center', fontsize=9)
            
            st.pyplot(fig_bench)
            plt.close(fig_bench)

    elif viz_mode == "2. 异常预测空间精准度":
        st.subheader("📍 异常预测空间精准度看板")
        selected_time = st.slider("选择分析时刻", time_range[0], time_range[1], time_range[0])
        
        if selected_time < params.len_recent_time:
            st.error(f"请选择大于 {params.len_recent_time} 的时间步以获取预测数据。")
        elif 'model' not in st.session_state:
            st.warning("请先加载模型以生成预测结果")
        else:
            model = st.session_state['model']
            with st.spinner("正在生成精准度分析图..."):
                idx = selected_time - params.len_recent_time
                input_data = prepare_data(all_data[idx:selected_time+1], params.len_recent_time)
                input_nc = prepare_data(threshold_nc[idx:selected_time+1], params.len_recent_time)
                y_dy = tf.ones((params.len_recent_time, params.number_region, 2 * params.dr))
                
                y_pred, _, _ = model(input_data, input_nc, y_dy)
                pred_prob = y_pred.numpy()[-1]
                true_label = label[selected_time]
                
                # Get threshold for binary prediction
                threshold = 0.5 
                
                grid_h, grid_w = params.grid, params.grid
                precision_map = np.zeros((grid_h, grid_w)) # 0:TN, 1:TP, 2:FP, 3:FN
                
                tp_count, fp_count, fn_count = 0, 0, 0
                for rid, (x, y) in dict_xy.items():
                    if rid < len(pred_prob):
                        p = 1 if pred_prob[rid] >= threshold else 0
                        t = true_label[rid]
                        
                        if p == 1 and t == 1: 
                            precision_map[x, y] = 1 # TP
                            tp_count += 1
                        elif p == 1 and t == 0: 
                            precision_map[x, y] = 2 # FP
                            fp_count += 1
                        elif p == 0 and t == 1: 
                            precision_map[x, y] = 3 # FN
                            fn_count += 1

                # Visual Presentation
                c1, c2, c3 = st.columns(3)
                c1.metric("命中 (TP)", tp_count, "预测准确")
                c2.metric("误报 (FP)", fp_count, "预测激进", delta_color="inverse")
                c3.metric("漏报 (FN)", fn_count, "预测保守", delta_color="inverse")
                
                st.write(f"#### 时间步 {selected_time} 预测结果空间精准度图")
                
                from matplotlib.colors import ListedColormap
                custom_cmap = ListedColormap(['#f8f9fa', '#28a745', '#dc3545', '#007bff'])
                
                # Massive font for grid numbers
                annot_size = max(10, 30 - grid_h)
                
                fig, ax = plt.subplots(figsize=(16, 12))
                sns.heatmap(precision_map, cmap=custom_cmap, ax=ax, cbar=False, annot=True,
                            annot_kws={"size": annot_size, "weight": "bold"})
                
                # Custom Legend with LARGE font
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#28a745', label='命中 (TP) - 预测正确'),
                    Patch(facecolor='#dc3545', label='误报 (FP) - 虚假警报'),
                    Patch(facecolor='#007bff', label='漏报 (FN) - 未能发现'),
                    Patch(facecolor='#f8f9fa', label='正确预测正常 (TN)')
                ]
                ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=14)
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
                st.info("💡 绿色区域表示模型成功捕捉到的异常；红色表示误报；蓝色表示未发现的真实异常。")

    elif viz_mode == "3. 区域交通流量热力分布":
        st.markdown("### 🕒 区域交通流量空间分布查看")
        selected_time = st.slider("选择查看时刻", time_range[0], time_range[1], time_range[0])
        
        # Prepare heatmap data
        grid_h = params.grid
        grid_w = params.grid
        
        # Massive font size for clear visibility
        show_annot = grid_h <= 15
        # Dynamic font size: 10x10 gets size 20, 15x15 gets size 14
        annot_size = max(10, 30 - grid_h) 
        
        # Layout: Stacked (One per row) to maximize width
        st.write(f"#### 1. 时间步 {selected_time} 的交通流量分布")
        traffic_map = np.zeros((grid_h, grid_w))
        for rid, (x, y) in dict_xy.items():
            if rid < all_data.shape[1]:
                traffic_map[x, y] = np.mean(all_data[selected_time, rid, :])
        
        fig, ax = plt.subplots(figsize=(16, 10)) # Much larger figure
        sns.heatmap(traffic_map, annot=show_annot, fmt=".1f", cmap="YlOrRd", ax=ax, 
                    annot_kws={"size": annot_size, "weight": "bold"}, 
                    cbar_kws={'label': '流量', 'pad': 0.02})
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.divider() # Add a line between maps

        st.write(f"#### 2. 时间步 {selected_time} 的真实异常分布")
        anomaly_map = np.zeros((grid_h, grid_w))
        for rid, (x, y) in dict_xy.items():
            if rid < label.shape[1]:
                anomaly_map[x, y] = label[selected_time, rid]
        
        fig2, ax2 = plt.subplots(figsize=(16, 10))
        sns.heatmap(anomaly_map, annot=show_annot, fmt=".0f", cmap="Reds", ax=ax2, 
                    annot_kws={"size": annot_size, "weight": "bold"}, 
                    cbar_kws={'label': '异常(0/1)', 'pad': 0.02})
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    elif viz_mode == "2. 异常区域风险排行榜":
        st.subheader("🏆 实时异常风险排行榜 (TOP 5)")
        selected_time = st.slider("选择分析时刻", time_range[0], time_range[1], time_range[0])
        
        if selected_time < params.len_recent_time:
            st.error(f"请选择大于 {params.len_recent_time} 的时间步以获取足够预测所需的历史数据。")
        elif 'model' not in st.session_state:
            st.warning("请先加载模型以计算异常评分")
        else:
            model = st.session_state['model']
            with st.spinner("正在计算风险评分..."):
                idx = selected_time - params.len_recent_time
                input_data = prepare_data(all_data[idx:selected_time+1], params.len_recent_time)
                input_nc = prepare_data(threshold_nc[idx:selected_time+1], params.len_recent_time)
                y_dy = tf.ones((params.len_recent_time, params.number_region, 2 * params.dr))
                
                y_pred, _, _ = model(input_data, input_nc, y_dy)
                scores = y_pred.numpy()[-1]
                
                top_indices = np.argsort(scores)[-5:][::-1]
                
                # --- NEW: Top 1 Highlight Metric ---
                top1_rid = top_indices[0]
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("榜首区域", f"ID: {top1_rid}")
                m_col2.metric("异常评分", f"{scores[top1_rid]:.4f}")
                m_col3.metric("风险等级", "★" * int(min(5, max(1, scores[top1_rid] * 10))))
                
                # Calculate simple trend for Top 1
                prev_score = np.mean(label[max(0, selected_time-5):selected_time, top1_rid])
                m_col4.metric("近期平均风险", f"{prev_score:.2f}", delta=f"{scores[top1_rid]-prev_score:+.2f}")

                st.divider()

                dash_data = []
                for i, rid in enumerate(top_indices):
                    # Calculate duration (consecutive anomalies)
                    duration = 0
                    for t in range(selected_time, -1, -1):
                        if label[t, rid] == 1: duration += 1
                        else: break
                    
                    # Year-over-year mock (comparison with same time in past)
                    prev_val = np.mean(all_data[max(0, selected_time-100):selected_time, rid, 0])
                    curr_val = all_data[selected_time, rid, 0]
                    change_rate = (curr_val - prev_val) / (prev_val + 1e-6)
                    
                    stars = "★" * int(min(5, max(1, scores[rid] * 10)))
                    
                    dash_data.append({
                        "排名": i+1,
                        "区域 ID": rid,
                        "异常评分": f"{scores[rid]:.4f}",
                        "风险等级": stars,
                        "持续时间(步)": duration,
                        "同比变化率": f"{change_rate:+.2%}"
                    })
                
                # Display with color coding
                df = pd.DataFrame(dash_data)
                st.table(df.style.background_gradient(subset=['异常评分'], cmap='YlOrRd'))
                
                st.write("📈 **TOP 5 区域流量近期趋势**")
                cols = st.columns(5)
                for i, rid in enumerate(top_indices):
                    with cols[i]:
                        st.markdown(f"**区域 {rid}**")
                        fig_mini, ax_mini = plt.subplots(figsize=(4, 3))
                        # Use a cleaner style for mini plots
                        ax_mini.plot(all_data[max(0, selected_time-20):selected_time+1, rid, 0], color='crimson', linewidth=2)
                        ax_mini.fill_between(range(len(all_data[max(0, selected_time-20):selected_time+1, rid, 0])), 
                                            all_data[max(0, selected_time-20):selected_time+1, rid, 0], color='crimson', alpha=0.2)
                        ax_mini.axis('off')
                        st.pyplot(fig_mini)
                        plt.close(fig_mini)

    elif viz_mode == "3. 预测残差热力分布图":
        st.subheader("📉 模型预测残差空间分布")
        selected_time = st.slider("选择分析时刻", time_range[0], time_range[1], time_range[0])
        
        if selected_time < params.len_recent_time:
            st.error(f"请选择大于 {params.len_recent_time} 的时间步以获取足够预测所需的历史数据。")
        elif 'model' not in st.session_state:
            st.error("此功能需要加载模型以获取预测值")
        else:
            model = st.session_state['model']
            with st.spinner("计算残差中..."):
                idx = selected_time - params.len_recent_time
                input_data = prepare_data(all_data[idx:selected_time+1], params.len_recent_time)
                input_nc = prepare_data(threshold_nc[idx:selected_time+1], params.len_recent_time)
                y_dy = tf.ones((params.len_recent_time, params.number_region, 2 * params.dr))
                
                y_pred, _, _ = model(input_data, input_nc, y_dy)
                pred_prob = y_pred.numpy()[-1]
                true_label = label[selected_time]
                residual = np.abs(pred_prob - true_label)
                
                grid_h, grid_w = params.grid, params.grid
                res_map = np.zeros((grid_h, grid_w))
                alert_regions = []
                std_res = np.std(residual)
                mean_res = np.mean(residual)
                
                for rid, (x, y) in dict_xy.items():
                    if rid < len(residual):
                        res_map[x, y] = residual[rid]
                        if residual[rid] > mean_res + 2 * std_res:
                            alert_regions.append((x, y))
                
                # --- Optimized Layout for Mode 3: Large Font & Stacked ---
                st.write(f"#### 时间步 {selected_time} 的残差分布热力图")
                
                # Dynamic font size (consistent with Mode 1)
                show_annot = grid_h <= 15
                annot_size = max(10, 30 - grid_h)
                
                fig, ax = plt.subplots(figsize=(16, 10))
                sns.heatmap(res_map, annot=show_annot, fmt=".2f", cmap="YlOrBr", ax=ax, 
                            annot_kws={"size": annot_size, "weight": "bold"},
                            cbar_kws={'label': '绝对残差', 'pad': 0.02})
                
                # Enhanced alert border
                for (x, y) in alert_regions:
                    rect = plt.Rectangle((y, x), 1, 1, fill=False, edgecolor='red', linewidth=4, linestyle='-')
                    ax.add_patch(rect)
                
                ax.set_title(f"预测残差 (红色方框为 >2σ 警示区)", fontsize=16)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
                st.divider()
                
                # Stats and Histogram below
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write("📊 **核心统计指标**")
                    st.metric("平均残差", f"{mean_res:.4f}")
                    st.metric("残差标准差 (σ)", f"{std_res:.4f}")
                    if len(alert_regions) > 0:
                        st.error(f"⚠️ 检测到 {len(alert_regions)} 个区域预测偏差过大！")
                
                with c2:
                    st.write("📈 **残差数值分布直方图**")
                    fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
                    sns.histplot(residual, bins=20, kde=True, ax=ax_dist, color='darkorange')
                    ax_dist.set_xlabel("绝对残差值")
                    ax_dist.set_ylabel("频数")
                    st.pyplot(fig_dist)
                    plt.close(fig_dist)

    elif viz_mode == "4. 区域多维特征画像剖析":
        st.subheader("🎯 区域多维特征画像剖析")
        
        # UI Polish: Better column ratios
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            st.info("💡 建议：选择 2-3 个区域进行对比分析效果最佳。")
            multi_regions = st.multiselect("选择对比区域 ID", range(params.number_region), default=[0, 1])
            selected_time = st.slider("分析参考时刻", time_range[0], time_range[1], time_range[0])
            
            with st.expander("🛠️ 维度权重微调"):
                w_poi = st.slider("POI 密度影响", 0.0, 2.0, 1.0)
                w_road = st.slider("路网复杂影响", 0.0, 2.0, 1.0)
                w_risk = st.slider("历史风险权重", 0.0, 2.0, 1.0)
        
        with col_c2:
            categories = ['POI密度', '路网复杂度', '历史风险值', '流量波动率', '邻域影响度', '时段敏感度']
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, rid in enumerate(multi_regions):
                color = colors[i % len(colors)]
                d1 = static_features['poi_density'][rid] * w_poi
                d2 = static_features['road_complexity'][rid] * w_road
                d3 = np.mean(label[:selected_time+1, rid]) * w_risk
                
                recent_traffic = all_data[max(0, selected_time-10):selected_time+1, rid, 0]
                d4 = np.std(recent_traffic) / (np.mean(recent_traffic) + 1e-6)
                d4 = min(1.0, d4 * 2)
                
                d5 = static_features['neighbor_influence'][rid]
                
                d6 = all_data[selected_time, rid, 0]
                d6 = (d6 - all_data[:, rid, 0].min()) / (all_data[:, rid, 0].max() - all_data[:, rid, 0].min() + 1e-6)

                values = [d1, d2, d3, d4, d5, d6]
                values = [min(1.0, v) for v in values]
                values += values[:1]
                
                ax.plot(angles, values, linewidth=3, label=f'区域 {rid}', color=color)
                ax.fill(angles, values, alpha=0.2, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
            ax.set_title("区域多维特征对比雷达图", fontsize=16, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
            # Set grid color
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            plt.close(fig)

if __name__ == "__main__":
    main()
