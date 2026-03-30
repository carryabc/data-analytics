# Data Analysis Web Application
# Run: streamlit run app.py --server.port 8501

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Data Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px 8px 0 0; }
.stMetric { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white; }
.stMetricValue { color: white !important; }
.stMetricLabel { color: rgba(255,255,255,0.8) !important; }
[data-testid="stMainBlockContainer"] { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ============ 工具函数 ============

def load_data(file) -> pd.DataFrame:
    """加载 CSV 或 Excel 文件"""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file, encoding='utf-8-sig')
        elif file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.error(f"文件读取失败: {e}")
    return None


def auto_clean(df: pd.DataFrame) -> pd.DataFrame:
    """自动数据清洗"""
    df = df.copy()
    original_rows = len(df)
    
    # 删除完全为空的行
    df = df.dropna(how='all')
    
    # 删除完全为空的列
    df = df.dropna(how='all', axis=1)
    
    # 去除重复行
    df = df.drop_duplicates()
    
    # 填充缺失值
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # 尝试转换日期列
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    cleaned_rows = len(df)
    return df, original_rows - cleaned_rows


def get_basic_stats(df: pd.DataFrame) -> dict:
    """获取基础统计信息"""
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'text_columns': len(df.select_dtypes(include=['object']).columns),
        'date_columns': len(df.select_dtypes(include=['datetime']).columns),
        'missing_cells': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    return stats


def get_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """获取数值列的详细统计"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().T


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """计算相关性矩阵"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return pd.DataFrame()
    return numeric_df.corr()


def get_category_freq(df: pd.DataFrame, col: str, top: int = 20) -> dict:
    """获取分类变量的频次"""
    if col not in df.columns:
        return {}
    return df[col].value_counts().head(top).to_dict()


# ============ 主应用 ============

def main():
    # 标题
    st.title("📊 Data Analytics Platform")
    st.markdown("---")
    
    # 初始化 session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    if 'auto_clean_done' not in st.session_state:
        st.session_state.auto_clean_done = False
    
    # ============ 侧边栏 ============
    with st.sidebar:
        st.header("📁 数据操作")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "上传 CSV 或 Excel 文件",
            type=['csv', 'xlsx', 'xls'],
            help="支持 .csv, .xlsx, .xls 格式"
        )
        
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.df_cleaned = df.copy()
                st.session_state.auto_clean_done = False
                st.success(f"✅ 文件已加载: {df.shape[0]} 行 × {df.shape[1]} 列")
        
        st.divider()
        
        # 一键清洗按钮
        st.subheader("🧹 数据清洗")
        if st.button("🔄 一键自动清洗", use_container_width=True, type="primary"):
            if st.session_state.df is not None:
                with st.spinner("正在清洗数据..."):
                    df_cleaned, removed = auto_clean(st.session_state.df)
                    st.session_state.df_cleaned = df_cleaned
                    st.session_state.auto_clean_done = True
                    st.success(f"✅ 清洗完成！删除了 {removed} 行脏数据")
                    st.rerun()
            else:
                st.warning("请先上传数据文件")
        
        # 下载清洗后的数据
        if st.session_state.df_cleaned is not None:
            csv = st.session_state.df_cleaned.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📥 下载清洗后的数据",
                csv,
                "cleaned_data.csv",
                "text/csv",
                use_container_width=True
            )
        
        st.divider()
        
        # 全流程按钮
        st.subheader("🚀 一键全流程")
        if st.button("执行清洗→分析→可视化", use_container_width=True, type="primary"):
            if st.session_state.df is not None:
                st.session_state.auto_clean_done = True
                st.session_state.df_cleaned, _ = auto_clean(st.session_state.df)
                st.success("✅ 全流程执行完成！")
                st.rerun()
            else:
                st.warning("请先上传数据文件")
    
    # ============ 主内容区 ============
    if st.session_state.df is None:
        # 初始界面
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 上传文件", "待上传")
        with col2:
            st.metric("📊 数据行数", "—")
        with col3:
            st.metric("📋 数据列数", "—")
        
        st.info("👈 请在左侧边栏上传 CSV 或 Excel 文件开始分析")
        
        # 功能介绍
        st.markdown("### ✨ 功能概览")
        cols = st.columns(3)
        features = [
            ("📤 文件上传", "支持 CSV/Excel，自动识别表头"),
            ("🧹 数据清洗", "一键删除空行空列、填充缺失值"),
            ("📈 数据分析", "统计摘要、相关性矩阵、频次分析"),
            ("📊 数据可视化", "5种交互式图表，支持动态选择"),
            ("🔄 全流程处理", "一键完成清洗→分析→可视化"),
            ("📥 数据导出", "下载清洗后数据和图表")
        ]
        for i, (title, desc) in enumerate(features):
            with cols[i % 3]:
                st.markdown(f"**{title}**")
                st.caption(desc)
        return
    
    df = st.session_state.df
    df_clean = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else df
    
    # 数据概览指标
    st.subheader("📋 数据概览")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("总行数", f"{len(df):,}")
    with col2:
        st.metric("总列数", f"{len(df.columns):,}")
    with col3:
        st.metric("数值列", f"{len(df.select_dtypes(include=[np.number]).columns)}")
    with col4:
        st.metric("缺失值", f"{df.isna().sum().sum():,}")
    with col5:
        st.metric("重复行", f"{df.duplicated().sum():,}")
    
    st.markdown("---")
    
    # Tab 页面
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 数据预览", 
        "🧹 清洗转换", 
        "📊 可视化",
        "📈 分析报告"
    ])
    
    # ============ Tab 1: 数据预览 ============
    with tab1:
        st.subheader("数据预览（前100行）")
        
        # 列信息
        with st.expander("📌 列信息", expanded=False):
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                missing = df[col].isna().sum()
                col_info.append({"列名": col, "数据类型": dtype, "缺失值": missing})
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)
        
        # 数据表格
        st.dataframe(df.head(100), use_container_width=True, height=400)
        
        # 显示统计信息
        with st.expander("📊 基础统计信息"):
            stats = get_numeric_stats(df)
            if not stats.empty:
                st.dataframe(stats, use_container_width=True)
            else:
                st.info("当前数据没有数值列")
    
    # ============ Tab 2: 清洗转换 ============
    with tab2:
        st.subheader("数据清洗与转换")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🔄 自动清洗")
            st.write("自动执行以下操作：")
            st.write("- 删除空行/空列")
            st.write("- 填充缺失值（数值用均值，类别用众数）")
            st.write("- 去除重复行")
            st.write("- 自动转换数据类型")
            
            if st.button("执行自动清洗", type="primary"):
                df_clean, removed = auto_clean(df)
                st.session_state.df_cleaned = df_clean
                st.session_state.auto_clean_done = True
                st.success(f"✅ 清洗完成！删除了 {removed} 行脏数据")
        
        with col2:
            st.markdown("### ✏️ 手动转换")
            
            # 选择列
            col_to_transform = st.selectbox("选择要转换的列", df.columns.tolist())
            
            if col_to_transform:
                transform_type = st.selectbox(
                    "转换操作",
                    ["无", "重命名列", "删除列", "数值归一化", "删除缺失值行"]
                )
                
                if transform_type == "重命名列":
                    new_name = st.text_input("新列名")
                    if st.button("执行重命名"):
                        if st.session_state.df_cleaned is None:
                            st.session_state.df_cleaned = df.copy()
                        st.session_state.df_cleaned = st.session_state.df_cleaned.rename(columns={col_to_transform: new_name})
                        st.success("✅ 重命名成功")
                        st.rerun()
                
                elif transform_type == "删除列":
                    if st.button("执行删除"):
                        if st.session_state.df_cleaned is None:
                            st.session_state.df_cleaned = df.copy()
                        st.session_state.df_cleaned = st.session_state.df_cleaned.drop(columns=[col_to_transform])
                        st.success("✅ 删除成功")
                        st.rerun()
                
                elif transform_type == "数值归一化":
                    if st.button("执行归一化"):
                        if st.session_state.df_cleaned is None:
                            st.session_state.df_cleaned = df.copy()
                        col_data = st.session_state.df_cleaned[col_to_transform]
                        if pd.api.types.is_numeric_dtype(col_data):
                            min_val = col_data.min()
                            max_val = col_data.max()
                            if max_val != min_val:
                                st.session_state.df_cleaned[col_to_transform] = (col_data - min_val) / (max_val - min_val)
                                st.success("✅ 归一化成功")
                            else:
                                st.warning("列值全部相同，无法归一化")
                        else:
                            st.error("只能对数值列归一化")
                        st.rerun()
                
                elif transform_type == "删除缺失值行":
                    if st.button("执行删除"):
                        if st.session_state.df_cleaned is None:
                            st.session_state.df_cleaned = df.copy()
                        before = len(st.session_state.df_cleaned)
                        st.session_state.df_cleaned = st.session_state.df_cleaned.dropna(subset=[col_to_transform])
                        after = len(st.session_state.df_cleaned)
                        st.success(f"✅ 删除了 {before - after} 行")
                        st.rerun()
        
        # 预览清洗后的数据
        if st.session_state.df_cleaned is not None:
            st.markdown("### 📋 清洗后数据预览")
            st.dataframe(st.session_state.df_cleaned.head(50), use_container_width=True, height=300)
    
    # ============ Tab 3: 可视化 ============
    with tab3:
        st.subheader("数据可视化")
        
        # 选择图表类型
        chart_type = st.selectbox(
            "选择图表类型",
            ["散点图", "柱状图", "折线图", "箱线图", "饼图"],
            index=0
        )
        
        # 获取可用列
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df_clean.columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if chart_type == "饼图":
                x_col = st.selectbox("选择分类列", all_cols)
                y_col = None
            else:
                x_col = st.selectbox("选择 X 轴", all_cols, index=0)
                y_col = st.selectbox("选择 Y 轴", numeric_cols if numeric_cols else all_cols)
        
        with col2:
            if chart_type != "饼图" and y_col:
                agg_func = st.selectbox("聚合方式", ["sum", "mean", "count", "first"])
            else:
                agg_func = "count"
        
        # 生成图表按钮
        if st.button("🎨 生成图表", type="primary"):
            try:
                if chart_type == "散点图":
                    if y_col and y_col in numeric_cols:
                        fig = px.scatter(df_clean, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                    else:
                        fig = px.scatter(df_clean, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                
                elif chart_type == "柱状图":
                    if y_col in numeric_cols:
                        grouped = df_clean.groupby(x_col)[y_col].agg(agg_func).reset_index()
                        fig = px.bar(grouped, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    else:
                        counts = df_clean[x_col].value_counts().reset_index()
                        counts.columns = [x_col, 'count']
                        fig = px.bar(counts, x=x_col, y='count', title=f"{x_col} 频次分布")
                
                elif chart_type == "折线图":
                    if y_col in numeric_cols:
                        grouped = df_clean.groupby(x_col)[y_col].agg(agg_func).reset_index()
                        fig = px.line(grouped, x=x_col, y=y_col, title=f"{y_col} by {x_col}", markers=True)
                    else:
                        st.error("Y 轴需要选择数值列")
                        fig = None
                
                elif chart_type == "箱线图":
                    if y_col in numeric_cols:
                        fig = px.box(df_clean, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    else:
                        st.error("Y 轴需要选择数值列")
                        fig = None
                
                elif chart_type == "饼图":
                    value_counts = df_clean[x_col].value_counts().reset_index()
                    value_counts.columns = [x_col, 'count']
                    fig = px.pie(value_counts, names=x_col, values='count', title=f"{x_col} 分布")
                
                if fig:
                    fig.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 下载按钮
                    img_data = fig.to_image(format="png", width=1200, height=600)
                    st.download_button(
                        "📥 下载图表 PNG",
                        img_data,
                        f"chart_{chart_type}.png",
                        "image/png",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"图表生成失败: {e}")
        
        # 快速图表预览
        st.markdown("### 🚀 快速图表预览")
        if numeric_cols and len(numeric_cols) >= 2:
            if st.button("📊 生成散点矩阵图"):
                fig = px.scatter_matrix(df_clean, dimensions=numeric_cols[:4], title="特征散点矩阵")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        if len(numeric_cols) >= 1:
            if st.button("📈 生成数值分布直方图"):
                fig = make_subplots(rows=2, cols=3, subplot_titles=numeric_cols[:6])
                for i, col in enumerate(numeric_cols[:6]):
                    row = i // 3 + 1
                    col_idx = i % 3 + 1
                    fig.add_trace(go.Histogram(x=df_clean[col], name=col, nbinsx=30), row=row, col=col_idx)
                fig.update_layout(height=500, showlegend=False, title_text="数值列分布")
                st.plotly_chart(fig, use_container_width=True)
    
    # ============ Tab 4: 分析报告 ============
    with tab4:
        st.subheader("📈 分析报告")
        
        # 基础统计
        stats = get_basic_stats(df_clean)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总行数", f"{stats['total_rows']:,}")
            st.metric("总列数", f"{stats['total_columns']:,}")
        with col2:
            st.metric("数值列", f"{stats['numeric_columns']}")
            st.metric("文本列", f"{stats['text_columns']}")
        with col3:
            st.metric("缺失值", f"{stats['missing_cells']:,}")
            st.metric("重复行", f"{stats['duplicate_rows']:,}")
        
        st.markdown("---")
        
        # 数值列统计
        st.markdown("### 📊 数值列统计摘要")
        numeric_stats = get_numeric_stats(df_clean)
        if not numeric_stats.empty:
            st.dataframe(numeric_stats, use_container_width=True)
        else:
            st.info("无数值列数据")
        
        st.markdown("---")
        
        # 相关性矩阵
        if len(numeric_stats) > 1:
            st.markdown("### 🔗 相关性矩阵")
            corr = get_correlation_matrix(df_clean)
            if not corr.empty:
                fig = px.imshow(
                    corr,
                    labels=dict(x="特征", y="特征", color="相关系数"),
                    x=corr.columns,
                    y=corr.index,
                    color_continuous_scale="RdBu_r",
                    title="特征相关性热力图"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # 下载相关性矩阵
                csv = corr.to_csv().encode('utf-8-sig')
                st.download_button("📥 下载相关性矩阵 CSV", csv, "correlation_matrix.csv", "text/csv")
        
        st.markdown("---")
        
        # 分类变量分析
        st.markdown("### 📋 分类变量频次分析")
        text_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            cat_col = st.selectbox("选择分类列进行分析", text_cols)
            if cat_col:
                freq = get_category_freq(df_clean, cat_col, top=15)
                if freq:
                    freq_df = pd.DataFrame({"类别": freq.keys(), "频次": freq.values()})
                    fig = px.bar(freq_df, x='类别', y='频次', title=f"{cat_col} 频次分布", color='频次', color_continuous_scale="Viridis")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # 生成报告摘要
        st.markdown("---")
        st.markdown("### 📝 分析报告摘要")
        report = f"""
        ## 数据分析报告
        
        ### 数据概况
        - **总记录数**: {stats['total_rows']:,} 行
        - **特征数量**: {stats['total_columns']:,} 列
        - **数值型特征**: {stats['numeric_columns']} 个
        - **文本型特征**: {stats['text_columns']} 个
        
        ### 数据质量
        - **缺失值**: {stats['missing_cells']:,} 个单元格
        - **重复行**: {stats['duplicate_rows']:,} 行
        """
        if not numeric_stats.empty:
            try:
                mean_col = numeric_stats['mean'].idxmax() if 'mean' in numeric_stats.columns else None
                std_col = numeric_stats['std'].idxmax() if 'std' in numeric_stats.columns else None
                if mean_col and mean_col in numeric_stats.index:
                    report += f"\n        ### 数值特征统计\n        - **均值最高的列**: {mean_col} ({numeric_stats.loc[mean_col, 'mean']:.2f})\n"
                if std_col and std_col in numeric_stats.index:
                    report += f"- **标准差最大的列**: {std_col} ({numeric_stats.loc[std_col, 'std']:.2f})\n"
            except Exception:
                pass
        
        st.markdown(report)
    
    # 页脚
    st.markdown("---")
    st.caption("📊 Data Analytics Platform | Built with Streamlit + Plotly")


if __name__ == "__main__":
    main()
