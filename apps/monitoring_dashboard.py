"""Read-only local dashboard for the Step 14 monitoring run records.

    python -m pip install streamlit
    python -m streamlit run apps/monitoring_dashboard.py

It trains nothing, changes no policy and never writes to the database. This is a
local research view: there is no authentication, authorisation or network
hardening, so it must not be exposed publicly and must not be pointed at real
customer records.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Running a file inside apps/ puts apps/ on the import path, not the project
# root, so the src package is added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring.runner import load_runs


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "artifacts" / "monitoring" / "runs.sqlite3"

st.set_page_config(page_title="信用风险研究监控", layout="wide")

st.title("信用风险研究监控看板")
st.caption(
    "研究原型：运行时刻不是申请日期；"
    "复核提示不等于模型失效，不触发实际授信行动。"
)

try:
    records = load_runs(DATABASE)
except FileNotFoundError:
    st.info("尚未发现运行记录，请先执行批次监控。")
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()

if not records:
    st.info("记录库为空。")
    st.stop()

selected_id = st.selectbox("选择运行记录", list(records))
record = records[selected_id]

st.subheader("运行与版本")
st.json(
    {
        key: record.get(key)
        for key in (
            "运行编号",
            "批次编号",
            "执行时刻",
            "运行状态",
            "版本",
        )
    }
)

if record.get("运行状态") == "失败":
    st.error(record.get("错误说明", "本次运行失败，详见运行记录库。"))
    st.stop()

col1, col2, col3 = st.columns(3)

col1.metric("当前样本数", record["当前样本数"])
col2.metric("无效值申请数", record["无效值申请数"])

approval_rate = record["策略监控"]["模拟通过率"]
col3.metric(
    "模拟通过率",
    "未配置" if approval_rate is None else f"{approval_rate:.2%}",
)

st.subheader("内部复核提示")
if record["复核提示"]:
    st.dataframe(
        pd.DataFrame(record["复核提示"]), use_container_width=True
    )
else:
    st.info(
        "当前规则没有生成提示；"
        "这不构成模型有效、公平或合规的证明。"
    )

st.subheader("分布诊断")
st.dataframe(
    pd.DataFrame(record["分布摘要"]), use_container_width=True
)

st.subheader("抽样解释一致性")
st.json(record["解释审计"])

st.subheader("固定策略监控")
st.json(record["策略监控"])

st.subheader("有标签诊断")
supervised = record["有标签诊断"]

if supervised is None:
    st.info("本次没有提供标签，未计算监督评价。")
else:
    st.json(supervised["校准摘要"])
    st.dataframe(
        pd.DataFrame(supervised["校准分箱"]), use_container_width=True
    )

st.caption(
    "这里只比较和展示单批记录。"
    "不同模型、参考分布或策略版本的指标，不应直接拼成同一趋势。"
)
