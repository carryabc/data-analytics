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
import re
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


# ============ 本地规则解析器（无需API） ============

def local_query_parser(df: pd.DataFrame, user_query: str) -> tuple:
    """
    本地规则解析器，无需API（智能增强版）
    返回: (结果, 操作描述, 错误信息)
    """
    if df is None or df.empty:
        return None, None, "数据为空，请先上传数据"
    
    query = user_query.strip().lower()
    
    # 获取可用列
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    def find_column(keywords, prefer_numeric=False):
        """根据关键词智能匹配列名"""
        if not keywords:
            return None
        cols = numeric_cols if prefer_numeric else all_cols
        # 先精确匹配
        for col in cols:
            col_lower = col.lower()
            for kw in keywords:
                if kw == col_lower:
                    return col
        # 再模糊匹配
        for col in cols:
            col_lower = col.lower()
            for kw in keywords:
                if kw in col_lower:
                    return col
        return None
    
    # ===== 智能列名提取 =====
    # 从查询中提取可能的列名关键词
    query_keywords = []
    # 常见关键词映射
    keyword_map = {
        '金额': ['金额', '钱', '消费', '订单金额', '总金额', '销售额', '收入', '价格'],
        '年龄': ['年龄', '岁'],
        '工资': ['工资', '薪资', '薪水'],
        '数量': ['数量', '销量', '销售量'],
        '成绩': ['成绩', '分数'],
        '姓名': ['姓名', '名字', '名称'],
        '部门': ['部门', '科室'],
        '城市': ['城市', '地区'],
        '产品': ['产品', '商品'],
        '类别': ['类别', '分类', '类型'],
        '会员': ['会员', '客户', '用户', 'id'],
        '状态': ['状态'],
    }
    
    # 检测查询中的关键词
    detected_col_keywords = []
    for col_type, keywords in keyword_map.items():
        for kw in keywords:
            if kw in query:
                detected_col_keywords.append(col_type)
                break
    

# ============ 自然语言查询（API版本） ============
    top_match = re.search(r'前(\d+)条', query)
    if top_match:
        n = int(top_match.group(1))
        return df.head(n), f"显示前{n}条", None
    
    tail_match = re.search(r'后(\d+)条', query)
    if tail_match:
        n = int(tail_match.group(1))
        return df.tail(n), f"显示后{n}条", None
    
    # ===== 3. 处理"显示所有"或"全部" =====
    if '所有' in query or '全部' in query:
        return df, "显示全部数据", None
    
    # ===== 4. 处理简单数值筛选 =====
    operators = [
        (r'大于(\d+\.?\d*)', '>'),
        (r'小于(\d+\.?\d*)', '<'),
        (r'>=(\d+\.?\d*)', '>='),
        (r'<=(\d+\.?\d*)', '<='),
    ]
    
    for kw in ['金额', '年龄', '工资', '价格', '销量', '成绩', '分数']:
        col = find_column([kw], prefer_numeric=True)
        if col:
            for pattern, op in operators:
                match = re.search(pattern, query)
                if match:
                    value = float(match.group(1))
                    try:
                        if op == '>':
                            result = df[df[col] > value]
                        elif op == '<':
                            result = df[df[col] < value]
                        elif op == '>=':
                            result = df[df[col] >= value]
                        else:
                            result = df[df[col] <= value]
                        return result, f"筛选 {col}{op}{value}", None
                    except Exception as e:
                        return None, None, f"筛选失败: {str(e)}"
    
    # ===== 5. 处理平均值/总和/计数 =====
    agg_func = None
    if '平均' in query or '均值' in query:
        agg_func = 'mean'
    elif any(kw in query for kw in ['总和', '总计', '总额', '总金额']):
        agg_func = 'sum'
    elif any(kw in query for kw in ['数量', '总数', 'count']):
        agg_func = 'count'
    
    if agg_func:
        target_col = None
        for kw in ['金额', '工资', '价格', '销量', '销售', '收入', '年龄', '成绩']:
            target_col = find_column([kw], prefer_numeric=True)
            if target_col:
                break
        
        if not target_col and numeric_cols:
            target_col = numeric_cols[0]
        
        if target_col:
            try:
                if agg_func == 'mean':
                    result = df[target_col].mean()
                elif agg_func == 'sum':
                    result = df[target_col].sum()
                else:
                    result = len(df)
            except Exception as e:
                return None, None, f"计算失败: {str(e)}"
            return result, f"{target_col}的{agg_func}", None
        
        if numeric_cols:
            try:
                if agg_func == 'mean':
                    result = df[numeric_cols].mean().mean()
                elif agg_func == 'sum':
                    result = df[numeric_cols].sum().sum()
                else:
                    result = len(df)
            except Exception as e:
                return None, None, f"计算失败: {str(e)}"
            return result, f"数值列的{agg_func}", None
    
    # ===== 6. 处理最大值/最小值 =====
    if any(kw in query for kw in ['最大', '最高', '最多']):
        target_col = None
        for kw in ['金额', '工资', '价格', '销量', '销售', '年龄', '成绩']:
            target_col = find_column([kw], prefer_numeric=True)
            if target_col:
                break
        if not target_col and numeric_cols:
            target_col = numeric_cols[0]
        if target_col:
            result = df[target_col].max()
            return result, f"{target_col}的最大值", None
    
    if any(kw in query for kw in ['最小', '最低', '最少']):
        target_col = None
        for kw in ['金额', '工资', '价格', '销量', '年龄', '成绩']:
            target_col = find_column([kw], prefer_numeric=True)
            if target_col:
                break
        if not target_col and numeric_cols:
            target_col = numeric_cols[0]
        if target_col:
            result = df[target_col].min()
            return result, f"{target_col}的最小值", None
    
    # ===== 7. 处理分组统计 =====
    group_col = None
    for col in text_cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in query.split()):
            if len(col) < 20:
                group_col = col
                break
    
    if group_col:
        if any(kw in query for kw in ['数量', '多少']):
            result = df.groupby(group_col).size().reset_index(name='数量')
            return result, f"按{group_col}分组计数", None
        else:
            result = df.groupby(group_col).size().reset_index(name='数量')
            return result, f"按{group_col}分组计数", None
    
    # ===== 8. 处理缺失值统计 =====
    if '缺失' in query or '空值' in query:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            result = missing.reset_index()
            result.columns = ['列名', '缺失数量']
            return result, "缺失值统计", None
        return 0, "无缺失值", None
    
    # ===== 9. 处理数据概览 =====
    if '概览' in query or '基本信息' in query:
        stats = {
            '总行数': len(df),
            '总列数': len(df.columns),
            '数值列数': len(numeric_cols),
            '文本列数': len(text_cols),
            '缺失值总数': df.isnull().sum().sum(),
        }
        result = pd.DataFrame(list(stats.items()), columns=['指标', '值'])
        return result, "数据概览", None
    
    # ===== 10. 默认返回前10条 =====
    return df.head(10), "默认显示前10条", None

# ============ 自然语言查询（API版本） ============

def natural_language_query(df: pd.DataFrame, user_query: str, api_key: str = None) -> tuple:
    """
    使用自然语言查询数据
    返回: (结果, 生成的代码, 错误信息)
    """
    if df is None or df.empty:
        return None, None, "数据为空，请先上传数据"
    
    # 获取列名信息
    columns_info = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        sample_values = df[col].dropna().head(3).tolist()
        columns_info.append(f"- {col}: {col_type}, 示例值: {sample_values}")
    
    columns_str = "\n".join(columns_info)
    columns_list = [col for col in df.columns]
    
    prompt = f"""你是一个数据分析助手。请将用户的自然语言问题转换为 pandas 代码。

数据集信息:
- 列名: {columns_list}
- 列详情:
{columns_str}
- 数据行数: {len(df)}

要求:
1. 只生成 pandas 代码，使用变量名 'df'
2. 不要生成任何说明文字，只返回可执行的 Python 代码
3. 代码必须安全，只能操作 df
4. 支持中文问题
5. 如果问题是统计类（如平均、总和），返回聚合值即可
6. 如果是筛选，返回筛选后的 DataFrame

用户问题: {user_query}

请只返回代码，不要返回其他内容。"""

    try:
        import openai
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个数据分析助手，只返回可执行的 pandas Python 代码。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        generated_code = response.choices[0].message.content.strip()
        # 清理代码
        generated_code = generated_code.replace('```python', '').replace('```', '').strip()
        
    except Exception as e:
        return None, None, f"API调用失败: {str(e)}"
    
    # 执行代码
    try:
        # 安全执行：限制命名空间
        safe_names = {
            'df': df,
            'pd': pd,
            'np': np,
            'result': None
        }
        
        # 清理危险代码
        dangerous_patterns = ['import', 'os', 'sys', 'eval', 'exec', 'open', '__']
        for pattern in dangerous_patterns:
            if pattern in generated_code:
                return None, generated_code, "代码包含不安全的内容"
        
        # 执行代码
        exec(generated_code, safe_names)
        result = safe_names.get('result')
        
        # 如果没有result变量，尝试找到返回值
        if result is None:
            # 尝试从代码中提取结果
            match = re.search(r'result\s*=\s*(.+)', generated_code)
            if match:
                result_expr = match.group(1).strip()
                result = eval(result_expr, {'df': df, 'pd': pd, 'np': np})
        
        return result, generated_code, None
        
    except Exception as e:
        return None, generated_code, f"代码执行失败: {str(e)}"


def format_result(result):
    """格式化查询结果"""
    if result is None:
        return {"type": "empty", "value": None}
    
    # 检查是否是 pandas Series（需要在 DataFrame 检查之前）
    if isinstance(result, pd.Series):
        return {"type": "series", "data": result.to_dict()}
    
    if isinstance(result, pd.DataFrame):
        if len(result) == 0:
            return {"type": "empty", "value": "数据为空"}
        if len(result) == 1 and len(result.columns) == 1:
            return {"type": "metric", "value": float(result.iloc[0, 0])}
        return {"type": "dataframe", "data": result}
    
    if isinstance(result, (int, float, np.integer, np.floating)):
        return {"type": "metric", "value": float(result)}
    
    # 支持 numpy array（如 unique() 返回）
    if hasattr(result, '__iter__') and hasattr(result, 'tolist'):
        try:
            return {"type": "array", "data": result.tolist()}
        except:
            pass
    
    return {"type": "unknown", "data": str(result)}


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
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
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
                st.session_state.query_history = []
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
        
        st.divider()
        
        # API Key 配置
        st.subheader("🔑 OpenAI API Key")
        api_key = st.text_input("输入 API Key", type="password", help="用于自然语言查询功能")
        if api_key:
            st.session_state.api_key = api_key
    
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
            ("📥 数据导出", "下载清洗后数据和图表"),
            ("💬 自然语言", "用中文提问，自动查询数据")
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 数据预览", 
        "🧹 清洗转换", 
        "📊 可视化",
        "📈 分析报告",
        "💬 自然语言查询"
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
                    try:
                        img_data = fig.to_image(format="png", width=1200, height=600)
                        st.download_button(
                            "📥 下载图表 PNG",
                            img_data,
                            f"chart_{chart_type}.png",
                            "image/png",
                            use_container_width=True
                        )
                    except Exception as e:
                        # 检查是否是 kaleido 相关错误
                        if 'kaleido' in str(e).lower():
                            st.warning("图表下载功能暂时不可用")
                        else:
                            st.info(f"图表下载暂时不可用: {str(e)}")
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
    
    # ============ Tab 5: 自然语言查询 ============
    with tab5:
        st.subheader("💬 自然语言查询（本地解析，无需API）")
        
        # 说明
        with st.expander("ℹ️ 使用说明", expanded=True):
            st.markdown("""
            **支持的查询类型（本地解析，无需API）：**
            - 条件筛选：`金额大于1000`、`年龄小于30`
            - 聚合统计：`平均金额`、`总和`、`最大值`
            - 排序/TOP：`前10条`、`后5条`
            - 分组聚合：`每个部门的数量`
            - 数据概览：`概览`、`缺失值统计`
            
            **示例问题：**
            - "显示前10条"
            - "金额大于1000"
            - "平均年龄"
            - "每个部门的数量"
            - "统计缺失值"
            """)
        
        # 查询输入（无需API Key）
        st.markdown("### 请输入您的问题")
        user_query = st.text_input(
            "例如：显示消费金额大于1000的记录",
            placeholder="输入您的自然语言问题...",
            label_visibility="collapsed"
        )
        
        col_btn, col_example = st.columns([1, 3])
        with col_btn:
            query_button = st.button("🔍 查询", type="primary", use_container_width=True)
        with col_example:
            example_queries = [
                "显示前10条",
                "金额大于1000",
                "平均年龄",
                "每个部门的数量",
                "统计缺失值"
            ]
            selected_example = st.selectbox("示例问题", [""] + example_queries, key="nl_example")
            if selected_example:
                user_query = selected_example
        
        # 执行查询（使用本地解析器）
        if query_button and user_query:
            with st.spinner("正在解析您的查询..."):
                result, explanation, error = local_query_parser(df_clean, user_query)
                
                if error:
                    st.error(error)
                else:
                    # 记录历史
                    result_type = "error"
                    if result is not None:
                        try:
                            result_type = format_result(result)['type']
                        except:
                            result_type = "dataframe"
                    st.session_state.query_history.append({
                        "query": user_query,
                        "explanation": explanation,
                        "result_type": result_type
                    })
                    
                    # 显示操作说明
                    st.info(f"📌 {explanation}")
                    
                    # 显示结果
                    formatted = format_result(result)
                    
                    if formatted['type'] == 'metric':
                        st.metric("查询结果", f"{formatted['value']:.2f}")
                    
                    elif formatted['type'] == 'dataframe':
                        st.markdown(f"**查询结果（共 {len(formatted['data'])} 条）:**")
                        st.dataframe(formatted['data'], use_container_width=True)
                        
                        # 如果结果适合可视化
                        if len(formatted['data']) > 0:
                            st.markdown("**可视化结果:**")
                            cols_for_chart = formatted['data'].select_dtypes(include=[np.number]).columns.tolist()
                            if cols_for_chart:
                                chart_col = st.selectbox("选择要可视化的列", cols_for_chart, key="nl_chart_col")
                                if chart_col:
                                    fig = px.bar(formatted['data'], x=formatted['data'].index, y=chart_col, title=f"{chart_col} 可视化")
                                    fig.update_layout(height=400, xaxis_title="", showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    elif formatted['type'] == 'series':
                        st.markdown("**查询结果:**")
                        result_df = pd.DataFrame({'类别': formatted['data'].keys(), '值': formatted['data'].values})
                        st.dataframe(result_df, use_container_width=True)
                        fig = px.bar(result_df, x='类别', y='值', title="结果可视化")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif formatted['type'] == 'array':
                        st.markdown("**唯一值列表:**")
                        st.write(formatted['data'])
        
        # 显示查询历史
        if st.session_state.query_history:
            st.markdown("---")
            st.markdown("### 📜 查询历史")
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"查询 {i+1}: {item['query'][:50]}...", expanded=False):
                    st.markdown(f"**问题:** {item['query']}")
                    if item.get('explanation'):
                        st.markdown(f"**操作:** {item['explanation']}")
        
        # 清空历史
        if st.button("🗑️ 清空查询历史"):
            st.session_state.query_history = []
            st.rerun()
    
    # 页脚
    st.markdown("---")
    st.caption("📊 Data Analytics Platform | Built with Streamlit + Plotly")


if __name__ == "__main__":
    main()
