# 🔬 Multi-Agent Research Assistant

A research tool that takes a query, searches the web, summarises findings, cross-validates claims, and produces a structured report — all automatically. Built as a 4-agent pipeline using LangGraph, with Groq for fast LLM inference and a Streamlit frontend.

## 🏗️ Architecture

```
Query → [🔍 SearchAgent] → [📝 SummarizeAgent] → [✅ FactCheckAgent] → [📄 ReportAgent] → Report
```

All four agents share a typed `ResearchState` (defined in `state.py`) that gets progressively enriched as it flows through the pipeline.

**🔍 SearchAgent** — Queries DuckDuckGo and pulls 6 live sources per query.

**📝 SummarizeAgent** — Synthesises all sources into a summary and extracts 5–8 verifiable factual claims.

**✅ FactCheckAgent** — Cross-validates each claim against the original sources. Tags them as *confirmed*, *conflicting*, or *unverified* with supporting evidence.

**📄 ReportAgent** — Generates a structured Markdown report with executive summary, findings, fact-check results, and citations.

## 📁 Project structure

```
├── app.py                  # Streamlit UI
├── graph.py                # LangGraph pipeline (state graph + edges)
├── state.py                # Typed ResearchState schema
├── agents/
│   ├── search_agent.py     # DuckDuckGo integration
│   ├── summarize_agent.py  # LLM summarisation + claim extraction
│   ├── fact_check_agent.py # Claim cross-validation
│   └── report_agent.py     # Markdown report generation
├── requirements.txt
└── .env.example
```

## 🚀 Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your Groq API key to .env (free at console.groq.com)
```

### ▶️ Run the web app
```bash
streamlit run app.py
```

### 💻 Run from CLI
```bash
export GROQ_API_KEY=your_key
python graph.py
```

## 📖 How to use

1. Enter your Groq API key in the sidebar (or set it as an env var)
2. Type a research query
3. Hit **Research** — the pipeline runs through all 4 agents
4. Browse results in the **Report**, **Sources**, **Fact-Check**, and **Raw Data** tabs
5. 📥 Download the report as Markdown with one click

## 🔧 Built with

LangGraph · LangChain · Groq (Llama 3.1 8B) · DuckDuckGo Search · Streamlit
