# Data Analytics Platform
# Run: streamlit run app.py --server.port 8501

import streamlit as st
import warnings
# 仅抑制已知无害的pandas/plotly警告，不全局屏蔽
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')

from config import APP_TITLE, APP_ICON, APP_LAYOUT
from ui.styles import inject_custom_css
from ui.sidebar import render_sidebar
from ui.pages import (
    render_data_preview, render_clean_transform,
    render_visualization, render_analysis_report, render_query_page,
    render_advanced_analysis,
)
from ui.components import show_empty_state, show_metric_cards
from analysis.statistics import get_basic_stats
from utils.validators import validate_dataframe
from utils.history import push_history


# ============ 页面配置 ============
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# 注入自定义CSS
inject_custom_css()


# ============ 主应用 ============
def main():
    # 初始化 session state
    defaults = {
        'df': None,
        'df_cleaned': None,
        'query_history': [],
        'history_stack': [],
        'future_stack': [],
        'last_file_name': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    try:
        # 渲染侧边栏
        result = render_sidebar()
        if not isinstance(result, tuple) or len(result) != 2:
            new_df, file_changed = None, False
        else:
            new_df, file_changed = result

        # 处理文件变更
        if file_changed and new_df is not None:
            st.session_state.df = new_df
            st.session_state.df_cleaned = new_df.copy()
            st.session_state.history_stack = []
            st.session_state.future_stack = []
            st.session_state.query_history = []
            st.success(f"✅ 已加载: {new_df.shape[0]:,} 行 × {new_df.shape[1]} 列")
            st.rerun()

        # 空状态引导
        if st.session_state.df is None:
            show_empty_state()
            return

        # 获取当前数据
        df = st.session_state.df
        df_clean = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else df

        if not validate_dataframe(df_clean):
            return

        # 主标题
        st.title("📊 Data Analytics Platform")

        # 数据概览指标
        stats = get_basic_stats(df_clean)
        show_metric_cards(stats)

        st.markdown("---")

        # 主内容区 Tab 页面
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 数据预览",
            "🧹 清洗转换",
            "📊 可视化",
            "📈 分析报告",
            "💬 智能查询",
            "🔬 高级分析",
        ])

        with tab1:
            render_data_preview(df_clean)

        with tab2:
            render_clean_transform(df_clean, push_history)

        with tab3:
            render_visualization(df_clean)

        with tab4:
            render_analysis_report(df_clean)

        with tab5:
            render_query_page(df_clean)

        with tab6:
            render_advanced_analysis(df_clean)

        # 页脚
        st.markdown("---")
        st.caption("📊 Data Analytics Platform | Built with Streamlit & Plotly")

    except Exception as e:
        st.error("应用运行出现异常，请刷新页面重试")
        st.exception(e)


if __name__ == "__main__":
    main()
