

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from collections import Counter
import re

# 页面配置
st.set_page_config(
    page_title="管理研究方法论课程分析仪表盘",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS美化
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .highlight-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #93C5FD;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #F59E0B;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .small-text {
        font-size: 0.85rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)


# 数据处理函数
def preprocess_data(df):
    """根据你的数据特点进行预处理"""
    df_clean = df.copy()

    # 1. 处理数值字段
    numeric_cols = ['学分', '学时', '课堂规模']
    for col in numeric_cols:
        if col in df_clean.columns:
            # 转换数据类型，处理空值和特殊值
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            # 用中位数填充缺失值
            median_val = df_clean[col].median() if not df_clean[col].isna().all() else 0
            df_clean[col] = df_clean[col].fillna(median_val)

    # 2. 处理权重字段（基于你的数据特点）
    def extract_weight_from_text(weight_text):
        """从权重文本中提取数值"""
        if pd.isna(weight_text) or weight_text in ['0', '无', '', ' ']:
            return 50, 50  # 默认值

        text = str(weight_text)

        # 处理 "40/60" 这种格式
        if '/' in text:
            parts = text.split('/')
            if len(parts) == 2:
                try:
                    usual = int(parts[0].strip())
                    final = int(parts[1].strip())
                    return usual, final
                except:
                    pass

        # 处理 "20/80" 这种格式
        if '/' in text:
            parts = text.split('/')
            if len(parts) == 2:
                try:
                    usual = int(parts[0].strip())
                    final = int(parts[1].strip())
                    return usual, final
                except:
                    pass

        # 处理 "60/40" 这种格式
        if '/' in text and text != '0':
            parts = text.split('/')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]), int(parts[1])

        return 50, 50  # 默认值

    # 提取权重
    weight_info = df_clean['平时/期末权重'].apply(extract_weight_from_text)
    df_clean['平时权重'] = [w[0] for w in weight_info]
    df_clean['期末权重'] = [w[1] for w in weight_info]

    # 3. 处理布尔字段
    bool_cols = ['是否翻转课堂', '是否有软件实操', '是否有开题报告', '是否有答辩']
    bool_mapping = {
        '是': '是', '有': '是', 'yes': '是', 'Yes': '是',
        '否': '否', '无': '否', 'no': '否', 'No': '否', '': '否'
    }

    for col in bool_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('否')
            df_clean[col] = df_clean[col].apply(
                lambda x: bool_mapping.get(str(x).strip(), '否')
            )

    # 4. 处理文本字段
    text_cols = ['特色做法', '核心教材', '软件工具', '考核内容']
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('未提供')
            # 替换0为"未提供"
            df_clean[col] = df_clean[col].replace({'0': '未提供', '无': '未提供'})

    # 5. 分类字段处理
    categorical_cols = ['教学模式', '面向层次']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('未知')

    # 6. 创建学时分层
    if '学时' in df_clean.columns:
        df_clean['学时分层'] = pd.cut(
            df_clean['学时'],
            bins=[0, 32, 48, 100],
            labels=['短学时(≤32)', '中学时(33-48)', '长学时(>48)'],
            right=False
        )

    return df_clean


# 数据加载函数
@st.cache_data
def load_data():
    """加载并处理数据"""
    try:
        # 读取Excel文件
        df = pd.read_excel("双一流高校课程开设情况.xlsx", sheet_name='Sheet1')

        # 清理列名（去除空格等）
        df.columns = df.columns.str.strip()

        # 预处理数据
        df = preprocess_data(df)

        return df
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return pd.DataFrame()


# 分析函数
def analyze_software_tools(df):
    """分析软件工具使用情况"""
    all_tools = []
    for tools in df['软件工具'].dropna():
        if tools != '未提供':
            # 分割多种工具
            for tool in str(tools).split(','):
                for t in tool.split('、'):
                    clean_tool = t.strip()
                    if clean_tool and clean_tool != '无':
                        all_tools.append(clean_tool)

    if not all_tools:
        return pd.DataFrame()

    # 统计工具使用频率
    tool_counts = Counter(all_tools)
    top_tools = tool_counts.most_common(20)

    tools_df = pd.DataFrame(top_tools, columns=['软件工具', '使用课程数'])

    # 标记机房已有软件
    lab_has = ['SPSS', 'Stata', 'Excel']
    tools_df['状态'] = tools_df['软件工具'].apply(
        lambda x: '机房已有' if any(lab_tool.lower() in str(x).lower() for lab_tool in lab_has) else '需补充'
    )

    return tools_df


def analyze_teaching_methods(df):
    """分析教学方法"""
    methods_data = []

    # 翻转课堂比例
    if '是否翻转课堂' in df.columns:
        flipped_ratio = (df['是否翻转课堂'] == '是').mean() * 100
        methods_data.append({'方法': '翻转课堂', '实施比例(%)': flipped_ratio})

    # 软件实操比例
    if '是否有软件实操' in df.columns:
        software_ratio = (df['是否有软件实操'] == '是').mean() * 100
        methods_data.append({'方法': '软件实操', '实施比例(%)': software_ratio})

    # 开题报告比例
    if '是否有开题报告' in df.columns:
        proposal_ratio = (df['是否有开题报告'] == '是').mean() * 100
        methods_data.append({'方法': '开题报告', '实施比例(%)': proposal_ratio})

    # 答辩比例
    if '是否有答辩' in df.columns:
        defense_ratio = (df['是否有答辩'] == '是').mean() * 100
        methods_data.append({'方法': '课程答辩', '实施比例(%)': defense_ratio})

    return pd.DataFrame(methods_data)


# 主应用
def main():
    # 标题
    st.markdown('<h1 class="main-header">📊 管理研究方法论课程</h1>', unsafe_allow_html=True)


    # 加载数据
    df = load_data()

    if df.empty:
        st.warning("请确保 '双一流高校课程开设情况.xlsx' 文件在当前目录，且包含名为 'Sheet1' 的工作表")
        return

    # 显示基本统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("调研高校数", df['高校名称'].nunique())
    with col2:
        avg_hours = df['学时'].mean()
        st.metric("平均学时", f"{avg_hours:.1f}")
    with col3:
        flipped_pct = (df['是否翻转课堂'] == '是').mean() * 100
        st.metric("翻转课堂比例", f"{flipped_pct:.1f}%")
    with col4:
        software_pct = (df['是否有软件实操'] == '是').mean() * 100
        st.metric("软件实操比例", f"{software_pct:.1f}%")

    st.markdown("---")

    # 侧边栏筛选器
    st.sidebar.header("🔍 数据筛选")

    # 高校筛选
    universities = sorted(df['高校名称'].dropna().unique())
    selected_unis = st.sidebar.multiselect(
        "选择高校",
        universities,
        default=universities
    )

    # 学时筛选
    if '学时' in df.columns:
        min_hours, max_hours = int(df['学时'].min()), int(df['学时'].max())
        hour_range = st.sidebar.slider(
            "学时范围",
            min_hours, max_hours,
            (min_hours, max_hours)
        )

    # 教学模式筛选
    if '教学模式' in df.columns:
        methods = df['教学模式'].dropna().unique()
        selected_methods = st.sidebar.multiselect(
            "教学模式",
            methods,
            default=list(methods)
        )

    # 应用筛选
    filtered_df = df.copy()
    if selected_unis:
        filtered_df = filtered_df[filtered_df['高校名称'].isin(selected_unis)]
    if '学时' in df.columns:
        filtered_df = filtered_df[(filtered_df['学时'] >= hour_range[0]) & (filtered_df['学时'] <= hour_range[1])]
    if '教学模式' in df.columns and selected_methods:
        filtered_df = filtered_df[filtered_df['教学模式'].isin(selected_methods)]
    #xinsheng
    st.sidebar.markdown("---")
    with st.sidebar.expander("★ 致新生的一封信", expanded=False):
        st.markdown("""
        <div style="text-indent: 2em; line-height: 1.6; font-size: 14px; color: #4B5563;">
        亲爱的同学们：

        欢迎踏上管理研究方法的学习之旅！作为一门连接理论与实践、思维与工具的核心课程，《管理研究方法论》不仅是学术研究的基石，更是未来职场竞争力的重要支撑。

        在这门课程中，你将不再是被动的知识接受者，而是主动的研究探索者。通过对比16所一流高校的教学实践，我们发现：成功的学习者往往具备三个特质——<strong>好奇心</strong>、<strong>执行力</strong>、<strong>协作力</strong>。

        32学时的课程虽然紧凑，但我们已经为你规划了清晰的路线图。记住，软件操作只是工具，研究思维才是核心。当你完成第一个数据分析、撰写第一篇研究方案时，那种创造的成就感将远超任何考试分数。

        让我们携手开启这段探索之旅，在学习中发现研究的乐趣，在挑战中收获成长的喜悦！
        </div>

        <div style="text-align: right; font-style: italic; margin-top: 15px; color: #6B7280;">
        —— 你的学长学姐们
        </div>
        """, unsafe_allow_html=True)

    # 创建标签页
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏫 课程概览",
        "🛠️ 软件工具",
        "📊 考核评估",
        "✨ 特色做法",
        "📋 详细数据",
        "💡 课程建议",
    ])

    # TAB 1: 课程概览
    with tab1:
        st.markdown('<h2 class="sub-header">🏫 课程基本信息分析</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # 学时分布
            if '学时分层' in filtered_df.columns:
                hour_dist = filtered_df['学时分层'].value_counts()
                fig1 = px.pie(
                    values=hour_dist.values,
                    names=hour_dist.index,
                    title='课程学时分布',
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    hole=0.4
                )
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)

            # 教学模式分布
            if '教学模式' in filtered_df.columns:
                mode_dist = filtered_df['教学模式'].value_counts()
                fig2 = px.bar(
                    x=mode_dist.index,
                    y=mode_dist.values,
                    title='教学模式分布',
                    labels={'x': '教学模式', 'y': '课程数'},
                    color=mode_dist.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig2, use_container_width=True)

        with col2:
            # 课堂规模分析
            if '课堂规模' in filtered_df.columns:
                fig3 = px.box(
                    filtered_df,
                    y='课堂规模',
                    title='课堂规模分布',
                    points='all'
                )
                fig3.update_layout(showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

            # 教学方法实施情况
            methods_df = analyze_teaching_methods(filtered_df)
            if not methods_df.empty:
                fig4 = px.bar(
                    methods_df,
                    x='方法',
                    y='实施比例(%)',
                    title='教学方法实施比例',
                    color='实施比例(%)',
                    color_continuous_scale='Teal',
                    text='实施比例(%)'
                )
                fig4.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                st.plotly_chart(fig4, use_container_width=True)

        # 短学时课程分析
        st.markdown("##### 🎯 短学时(≤32)课程特点分析")
        short_hour_courses = filtered_df[filtered_df['学时'] <= 32]

        if not short_hour_courses.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("短学时课程数", len(short_hour_courses))
            with col2:
                avg_credit = short_hour_courses['学分'].mean()
                st.metric("平均学分", f"{avg_credit:.1f}")
            with col3:
                flipped_ratio = (short_hour_courses['是否翻转课堂'] == '是').mean() * 100
                st.metric("翻转课堂比例", f"{flipped_ratio:.1f}%")
            with col4:
                software_ratio = (short_hour_courses['是否有软件实操'] == '是').mean() * 100
                st.metric("软件实操比例", f"{software_ratio:.1f}%")

            # 短学时课程应对策略
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**💡 短学时课程应对策略建议：**")
            st.markdown("""
            1. **课前准备**：提前阅读教材1-3章，安装所需软件
            2. **重点突出**：聚焦研究方法核心模块
            3. **项目驱动**：用小项目贯穿学习全过程
            4. **混合学习**：线上资源辅助课堂教学
            5. **小组协作**：分组完成研究设计任务
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: 软件工具
    with tab2:

        st.markdown('<h2 class="sub-header">🛠️ 软件工具使用分析</h2>', unsafe_allow_html=True)

        # 软件工具分析
        tools_df = analyze_software_tools(filtered_df)

        if not tools_df.empty:
            col1, col2 = st.columns([3, 1])

            with col1:
                # 软件使用频率
                fig_tools = px.bar(
                    tools_df,
                    x='软件工具',
                    y='使用课程数',
                    color='状态',
                    title='软件工具使用情况',
                    color_discrete_map={'机房已有': '#10B981', '需补充': '#3B82F6'},
                    text='使用课程数'
                )
                fig_tools.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_tools, use_container_width=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**📊 软件使用统计**")

                total_courses = len(filtered_df)
                software_courses = (filtered_df['是否有软件实操'] == '是').sum()
                st.metric("开设软件课程", f"{software_courses}/{total_courses}")

                st.markdown("**💡 学习建议：**")
                st.markdown("""
                1. **SPSS** - 必学（7门课程使用）
                2. **Stata** - 重点（4门课程使用）
                3. **AI工具** - 新兴（2门课程使用）
                4. **Python** - 进阶（自主补充）
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("暂无软件工具使用数据")

        # 软件实操课程分析
        st.markdown("##### 💻 软件实操课程特点")

        software_courses = filtered_df[filtered_df['是否有软件实操'] == '是']

        if not software_courses.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_hours = software_courses['学时'].mean()
                st.metric("平均学时", f"{avg_hours:.1f}")
            with col2:
                software_tools = software_courses['软件工具'].tolist()
                unique_tools = set()
                for tools in software_tools:
                    if tools != '未提供':
                        for tool in str(tools).split(','):
                            for t in tool.split('、'):
                                clean_tool = t.strip()
                                if clean_tool and clean_tool != '无':
                                    unique_tools.add(clean_tool)
                st.metric("软件种类", len(unique_tools))
            with col3:
                flipped_ratio = (software_courses['是否翻转课堂'] == '是').mean() * 100
                st.metric("翻转课堂比例", f"{flipped_ratio:.1f}%")

            # 显示软件课程列表
            with st.expander("📋 查看开设软件实操的课程"):
                for _, row in software_courses.iterrows():
                    st.markdown(f"**{row['高校名称']}** - {row['课程名']}")
                    st.markdown(f"软件工具：{row['软件工具']}")
                    st.markdown(f"学时：{int(row['学时'])} | 教学模式：{row['教学模式']}")
                    st.markdown("---")

    # TAB 3: 考核评估
    with tab3:
        st.markdown('<h2 class="sub-header">📊 考核评估方式分析</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # 考核权重分布
            if '平时权重' in filtered_df.columns and '期末权重' in filtered_df.columns:
                # 创建散点图
                fig_weight = px.scatter(
                    filtered_df,
                    x='平时权重',
                    y='期末权重',
                    title='考核权重分布',
                    labels={'平时权重': '平时成绩权重(%)', '期末权重': '期末成绩权重(%)'},
                    hover_data=['高校名称', '课程名', '学时'],
                    color='学时',
                    size='学时',
                    size_max=20,
                    color_continuous_scale='Viridis'
                )

                # 添加对角线
                fig_weight.add_shape(
                    type="line",
                    x0=0, y0=100, x1=100, y1=0,
                    line=dict(color="Red", width=2, dash="dash")
                )

                st.plotly_chart(fig_weight, use_container_width=True)

        with col2:
            # 考核方式统计
            assessment_methods = []
            for content in filtered_df['考核内容'].dropna():
                if content not in ['未提供', '无', '']:
                    assessment_methods.append(content.strip())

            if assessment_methods:
                method_counts = Counter(assessment_methods)
                common_methods = method_counts.most_common(10)

                if common_methods:
                    methods_df = pd.DataFrame(common_methods, columns=['考核方式', '频次'])

                    fig_methods = px.bar(
                        methods_df,
                        x='考核方式',
                        y='频次',
                        title='常见考核方式',
                        color='频次',
                        color_continuous_scale='RdBu',
                        text='频次'
                    )
                    fig_methods.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_methods, use_container_width=True)

        # 考核权重建议
        st.markdown("##### 🎯 本校考核权重设计建议")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📝 基于数据分析的建议：**")
            st.markdown("""
            | 考核环节 | 建议权重 | 说明 |
            |---------|---------|------|
            | 平时成绩 | 40% | 出勤、作业、课堂参与 |
            | 软件实操 | 25% | Stata/SPSS数据分析 |
            | 开题报告 | 15% | 研究设计方案 |
            | 期末论文 | 20% | 完整研究报告 |
            """, unsafe_allow_html=True)
            st.markdown("**总分：100%**")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**⚖️ 权重设计原则：**")
            st.markdown("""
            1. **过程导向**：强调平时积累(40%)
            2. **能力导向**：突出软件实操(25%)
            3. **实践导向**：重视研究设计(15%)
            4. **成果导向**：检验综合能力(20%)

            **📈 数据支持：**
            - 平均平时权重：43.8%
            - 软件实操课程：56.3%
            - 开题报告：31.3%
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # TAB 4: 特色做法
    with tab4:
        st.markdown('<h2 class="sub-header">✨ 各校特色做法与创新</h2>', unsafe_allow_html=True)

        # 筛选有特色做法的课程
        special_courses = filtered_df[filtered_df['特色做法'] != '未提供']

        if not special_courses.empty:
            # 分类展示特色做法
            categories = {
                "👥 小组协作": ["小组汇报", "案例分析", "小组讨论"],
                "🎯 实践导向": ["软件实操", "数据收集", "论文撰写"],
                "🤖 技术创新": ["AI", "智能体", "在线平台"],
                "👨‍🏫 专家分享": ["专家讲座", "学长分享", "跨专业交流"]
            }

            for category, keywords in categories.items():
                # 查找相关课程
                related_courses = []
                for _, row in special_courses.iterrows():
                    if any(keyword in str(row['特色做法']) for keyword in keywords):
                        related_courses.append(row)

                if related_courses:
                    st.markdown(f"##### {category}")
                    for course in related_courses[:3]:  # 显示前3个
                        with st.expander(f"**{course['高校名称']}** - {course['课程名']}"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**特色做法：** {course['特色做法']}")
                            with col2:
                                st.markdown(f"**学时：** {int(course['学时'])}")
                                st.markdown(f"**模式：** {course['教学模式']}")

            # 所有特色做法展示
            st.markdown("##### 📋 全部特色做法列表")
            for idx, row in special_courses.iterrows():
                with st.expander(f"{row['高校名称']} - {row['课程名']} ({int(row['学时'])}学时)"):
                    st.markdown(f"**特色做法：** {row['特色做法']}")
                    if row['软件工具'] != '未提供':
                        st.markdown(f"**软件工具：** {row['软件工具']}")
                    if row['考核内容'] != '未提供':
                        st.markdown(f"**考核方式：** {row['考核内容']}")
        else:
            st.info("暂无特色做法数据")

        # 可移植经验总结
        st.markdown("##### 💡 可移植的优秀经验")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        **基于数据分析的可移植经验：**

        1. **混合教学模式**（北京邮电大学）
           - 线上智能体辅助 + 线下项目式教学
           - 适合：软件实操课程

        2. **专家分享机制**（中国农业大学）
           - 邀请专家、学长进行案例分享
           - 适合：前沿方法介绍

        3. **全过程研究训练**（北京外国语大学）
           - 文献综述 → 问卷设计 → 数据分析 → 论文撰写
           - 适合：研究能力培养

        4. **小组协作学习**（多所高校）
           - 小组汇报 + 案例分析 + 项目合作
           - 适合：综合能力提升
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 5: 详细数据
    with tab5:
        st.markdown('<h2 class="sub-header">📋 详细数据浏览与导出</h2>', unsafe_allow_html=True)

        # 搜索功能
        search_term = st.text_input("🔍 搜索数据（高校、课程、软件等）", "")

        # 显示数据
        display_df = filtered_df.copy()

        if search_term:
            # 在文本列中搜索
            text_cols = ['高校名称', '课程名', '特色做法', '核心教材', '软件工具', '考核内容']
            mask = pd.Series([False] * len(display_df))
            for col in text_cols:
                if col in display_df.columns:
                    mask = mask | display_df[col].astype(str).str.contains(search_term, case=False, na=False)
            display_df = display_df[mask]

        # 选择显示的列
        default_cols = ['高校名称', '课程名', '学时', '学分', '教学模式', '是否翻转课堂',
                        '软件工具', '平时权重', '期末权重', '考核内容']

        available_cols = [col for col in default_cols if col in display_df.columns]
        selected_cols = st.multiselect(
            "选择显示的列",
            display_df.columns.tolist(),
            default=available_cols
        )

        if selected_cols:
            display_data = display_df[selected_cols]
        else:
            display_data = display_df

        # 显示数据表
        st.dataframe(
            display_data,
            use_container_width=True,
            height=600,
            column_config={
                "高校名称": st.column_config.TextColumn(width="medium"),
                "课程名": st.column_config.TextColumn(width="large"),
                "特色做法": st.column_config.TextColumn(width="medium"),
                "软件工具": st.column_config.TextColumn(width="medium"),
                "考核内容": st.column_config.TextColumn(width="medium")
            }
        )

        # 数据统计
        st.markdown("##### 📈 数据统计摘要")

        if not display_data.empty:
            stats_cols = st.columns(4)

            with stats_cols[0]:
                st.metric("显示记录数", len(display_data))
            with stats_cols[1]:
                if '学时' in display_data.columns:
                    avg_hours = display_data['学时'].mean()
                    st.metric("平均学时", f"{avg_hours:.1f}")
            with stats_cols[2]:
                if '平时权重' in display_data.columns:
                    avg_usual = display_data['平时权重'].mean()
                    st.metric("平时权重均值", f"{avg_usual:.1f}%")
            with stats_cols[3]:
                if '期末权重' in display_data.columns:
                    avg_final = display_data['期末权重'].mean()
                    st.metric("期末权重均值", f"{avg_final:.1f}%")

        # 数据下载
        st.markdown("##### 💾 数据导出")

        csv_data = display_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载当前数据 (CSV)",
            data=csv_data,
            file_name=f"管理研究方法论_课程数据_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    # TAB 6: 课程建议
    with tab6:
        st.markdown('<h2 class="sub-header">🎯 新生学习全攻略</h2>', unsafe_allow_html=True)

        # 创建三列布局
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # 16周课程安排
            st.markdown("##### 📅 16周详细课程安排")

            # 创建课程安排表格
            schedule_data = {
                "周次": ["1-2周", "3-4周", "5-6周", "7-8周", "9-10周", "11-12周", "13-14周", "15-16周"],
                "教学模块": [
                    "课程导论与研究方法基础",
                    "研究设计与问题提出",
                    "文献综述与理论框架",
                    "定量研究方法（SPSS/Stata）",
                    "质性研究方法",
                    "数据收集与处理实践",
                    "研究论文撰写指导",
                    "成果展示与课程总结"
                ],
                "核心任务": [
                    "掌握研究基本范式，安装软件",
                    "确定研究选题，设计研究方案",
                    "完成文献综述，建立理论框架",
                    "掌握描述统计、相关分析、回归分析",
                    "学习案例研究、访谈法、内容分析",
                    "设计问卷/实验，收集处理数据",
                    "撰写完整研究论文（8000字）",
                    "小组答辩，提交最终成果"
                ],
                "关键产出": [
                    "研究兴趣报告",
                    "开题报告框架",
                    "文献综述初稿",
                    "数据分析练习1-3",
                    "质性分析报告",
                    "数据集+处理文档",
                    "论文初稿",
                    "最终论文+答辩PPT"
                ]
            }

            schedule_df = pd.DataFrame(schedule_data)
            st.dataframe(
                schedule_df,
                use_container_width=True,
                height=400,
                column_config={
                    "周次": st.column_config.TextColumn(width="small"),
                    "教学模块": st.column_config.TextColumn(width="medium"),
                    "核心任务": st.column_config.TextColumn(width="large"),
                    "关键产出": st.column_config.TextColumn(width="medium")
                },
                hide_index=True
            )



        with col2:
            # 预习清单
            st.markdown("##### 📋 开学前预习清单")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**✅ 开学前必做事项：**")

            # 使用checkbox创建清单
            checklist_items = [
                ("购买李怀祖《管理研究方法论》教材", True),
                ("安装SPSS软件（官网下载试用版）", True),
                ("安装Stata软件（学校提供教育版）", True),
                ("预习教材第1-2章（研究方法基础）", True),
                ("思考2-3个潜在研究问题", True),
                ("准备移动硬盘/U盘（备份数据）", True)
            ]

            for item, checked in checklist_items:
                if checked:
                    st.markdown(f"✓ **{item}**")
                else:
                    st.markdown(f"□ {item}")

            st.markdown("---")



        with col3:

            # 提分策略
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            st.markdown("**💡 提分黄金策略：**")
            st.markdown("""
            1. **提前沟通**：与老师讨论研究选题
            2. **过程记录**：保留所有中间文件
            3. **规范先行**：严格遵循格式要求
            4. **团队协作**：发挥小组成员优势
            5. **迭代改进**：根据反馈持续优化
            """)
            st.markdown('</div>', unsafe_allow_html=True)


        col_a, col_b = st.columns(2)



        # 快速入门指南
        st.markdown("##### 🚀 快速入门三步曲")

        quick_guide_col1, quick_guide_col2, quick_guide_col3 = st.columns(3)

        with quick_guide_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📘 第一步：理论准备（第1-2周）**")
            st.markdown("""
            **目标**：建立方法论框架

            **行动清单**：
            - 精读教材1-3章
            - 整理关键概念
            - 确定研究兴趣方向
            - 完成第一次作业
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with quick_guide_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**💻 第二步：技能准备（第3-4周）**")
            st.markdown("""
            **目标**：掌握核心软件

            **行动清单**：
            - 完成SPSS基础教程
            - 掌握Stata基本命令
            - 处理第一个数据集
            - 提交数据分析练习
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with quick_guide_col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📝 第三步：研究设计（第5-6周）**")
            st.markdown("""
            **目标**：形成研究方案

            **行动清单**：
            - 确定研究选题
            - 设计研究方案
            - 完成开题报告
            - 组建研究小组
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    # 页脚信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
        <p>📊 管理研究方法论课程对比分析 | 基于17所双一流高校数据</p>
        <p>💡 数据来源：课程调研 | 分析时间：""" + datetime.now().strftime("%Y年%m月%d日") + """</p>
        <p>🎯 适配本校32学时课程 | 为新生提供选课与学习指导</p>
    </div>
    """, unsafe_allow_html=True)

# 运行应用
if __name__ == "__main__":
    main()
