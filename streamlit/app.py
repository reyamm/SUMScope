"""SUMScore: CPU-only replay of stored BRIGHT retrieval model scores."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SUMScore | Retrieval Evaluation", page_icon="🔎", layout="wide")
ROOT = Path(__file__).resolve().parent
# Locate experiment artifacts.
HOME = Path.home()
HUMAIN = HOME / "Desktop" / "Humain"
CANDIDATES = [ROOT / "data", HUMAIN / "q50", HUMAIN / "Q50", HUMAIN, ROOT]
DATA_DIR = next((p for p in CANDIDATES if (p / "probes_earth_science.json").exists()), CANDIDATES[0])
SUMMARY_DIRS = [HUMAIN / "drive-download-20261031T123949Z-1-001", HUMAIN, DATA_DIR, ROOT / "data"]

def find_summary(domain):
    for folder in SUMMARY_DIRS:
        p = folder / f"summaries_{domain}.json"
        if p.exists(): return p
    return None

NEG = -1e9

@st.cache_data(show_spinner=False)
def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def load_cost():
    p = DATA_DIR / "table_cost.csv"
    return pd.read_csv(p).set_index("arm") if p.exists() else None

def best(family):
    return max((float(x[0]) for x in family), default=NEG)

def rank_score(entry, method):
    if method == "SUMScore":
        return best(entry.get("L", [])) + best(entry.get("G", [])) if (entry.get("L") or entry.get("G")) else 2 * NEG
    if method == "Periscope":
        return best(entry.get("L", [])) + best(entry.get("S", [])) if (entry.get("L") or entry.get("S")) else 2 * NEG
    if method == "Rows only":
        return best(entry.get("L", []))
    if method == "Summary only":
        return best(entry.get("G", []))
    raise ValueError(method)

def online_cost(entry, method):
    families = {
        "SUMScore": ["L", "G"] if entry.get("summarized", False) else ["L"],
        "Periscope": ["L", "S"],
        "Rows only": ["L"],
        "Summary only": ["G"],
    }[method]
    return (sum(len(entry.get(f, [])) for f in families),
            sum(int(x[1]) for f in families for x in entry.get(f, [])))

def make_ranking(probes, qid, method, original_order=None):
    entries = {doc: v[qid] for doc,v in probes.items() if qid in v}
    # Preserve candidate order for equal scores.
    order = list(entries)
    if original_order is not None:
        order = [d for d in original_order if d in entries] + [d for d in order if d not in original_order]
    else:
        order.sort()
    order.sort(key=lambda d: -rank_score(entries[d], method))
    rows = [{"Rank": i, "Document": doc, "Relevance score": round(rank_score(entries[doc], method),3),
             "LLM calls":online_cost(entries[doc],method)[0],
             "Prompt tokens":online_cost(entries[doc],method)[1]} for i,doc in enumerate(order,1)]
    return pd.DataFrame(rows),entries

st.markdown("# SUMScore")
st.caption("Reasoning-aware retrieval | Evaluation dashboard")

st.caption("Evaluation replay · Recorded Qwen inference · No GPU required")

available = {d:(DATA_DIR/f"probes_{d}.json") for d in ["earth_science","biology","psychology","pony"] if (DATA_DIR/f"probes_{d}.json").exists()}
if not available:
    st.error(f"No probes JSON found in `{DATA_DIR}`. Check that Desktop/Humain/q50 contains probes_earth_science.json.")
    st.stop()

tabs = st.tabs(["Query replay", "Benchmark", "Methodology"])
with tabs[0]:
    left,right = st.columns([1,2.1],gap="large")
    with left:
        domain = st.selectbox("Select domain",list(available),format_func=lambda x:x.replace("_"," ").title())
        probes = load_json(str(available[domain]))
        qids = sorted({str(q) for v in probes.values() for q in v},key=lambda q:(not q.isdigit(),int(q) if q.isdigit() else q))
        qid = st.selectbox("Select saved question ID",qids, index=qids.index('3') if '3' in qids else 0)
        qfile = DATA_DIR/f"queries_{domain}.json"
        qmap = load_json(str(qfile)) if qfile.exists() else {}
        if qid in qmap:
            st.markdown("**Original BRIGHT question**")
            st.write(qmap[qid])
        else:
            st.caption("Question text unavailable; results are indexed by benchmark query ID.")
        n = st.slider("Show top documents",3,20,5)
    with right:
        embed_path = DATA_DIR/f"embed_{domain}.json"
        embed = load_json(str(embed_path)) if embed_path.exists() else {}
        # Recover initial retrieval order when available.
        source_scores = embed.get(qid,{}) if isinstance(embed,dict) else {}
        candidate_order = sorted(source_scores,key=lambda d:-float(source_scores[d])) if source_scores else None
        s_df, entries = make_ranking(probes,qid,"SUMScore",candidate_order)
        p_df,_ = make_ranking(probes,qid,"Periscope",candidate_order)
        a,b = st.columns(2)
        sum_calls,sum_tokens = (int(s_df['LLM calls'].sum()),int(s_df['Prompt tokens'].sum()))
        per_calls,per_tokens = (int(p_df['LLM calls'].sum()),int(p_df['Prompt tokens'].sum()))
        with a: st.metric("Fewer online LLM calls",f"{100*(1-sum_calls/per_calls):.1f}%" if per_calls else "N/A",f"{per_calls:,} → {sum_calls:,}")
        with b: st.metric("Fewer prompt tokens",f"{100*(1-sum_tokens/per_tokens):.1f}%" if per_tokens else "N/A",f"{per_tokens:,} → {sum_tokens:,}")
        st.caption(f"Saved query ID {qid} · {len(entries)} scored candidate documents · first-stage order used for tie-breaks when available")
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("SUMScore")
            st.dataframe(s_df.head(n)[["Rank","Document","Relevance score"]],use_container_width=True,hide_index=True)
        with c2:
            st.subheader("Periscope")
            st.dataframe(p_df.head(n)[["Rank","Document","Relevance score"]],use_container_width=True,hide_index=True)
        agree = s_df.iloc[0]['Document']==p_df.iloc[0]['Document']
        st.write("**Same top-ranked document:**", "Yes" if agree else "No")
        st.caption("Relevance scores are model log-odds, not probabilities. Ranking agreement does not establish retrieval accuracy.")
        st.download_button("Download this replay's rankings (CSV)",pd.concat([s_df.assign(Method="SUMScore"),p_df.assign(Method="Periscope")]).to_csv(index=False),file_name=f"sumscore_{domain}_q{qid}.csv",mime="text/csv")
        with st.expander("View cached summary of a ranked document"):
            summary_path = find_summary(domain)
            if summary_path:
                summaries = load_json(str(summary_path))
                found = [doc for doc in s_df["Document"] if doc in summaries]
                if found:
                    sd = st.selectbox("Summarized document", found, key="summary_doc")
                    st.write(summaries[sd].get("summary", "No summary text stored"))
                    st.caption("Offline summary used in the recorded evaluation.")
                else: st.caption("No cached summary is available for this selection.")
            else: st.caption("Summary cache is unavailable.")
        with st.expander("Inspect saved model evidence for a document"):
            d=st.selectbox("Document", list(s_df['Document']),key="inspect_doc")
            e=entries[d]
            has_summary = bool(e.get("summarized", False))
            st.write("**Document summary:**", "Generated and cached" if has_summary else "Not generated — short-document fallback")
            st.write("**Local passages scored:**", len(e.get("L", [])))
            st.write("**Strided passages scored:**", len(e.get("S", [])))
            st.write("**Local relevance:**", round(best(e.get("L", [])), 3))
            st.write("**Global relevance:**", round(best(e.get("S", [])), 3))
            if has_summary:
                st.write("**Summary relevance:**", round(best(e.get("G", [])), 3))
            else:
                st.caption("The recorded summary-score field uses a fallback for this document; no document summary was generated.")
with tabs[1]:
    st.subheader("Recorded full benchmark costs")
    cost = load_cost()
    if cost is not None:
        label_map={"embed":"Embedding","rows":"Rows only","periscope":"Periscope","rows+sum (prepend)":"Summary prepend","rows+sum (vote)":"SUMScore","summary only":"Summary only"}
        view=cost.reset_index();view['Method']=view['arm'].map(label_map)
        st.dataframe(view[['Method','LLM calls / query','prompt tokens / query (k)','tokens vs periscope']],hide_index=True,use_container_width=True)
        st.bar_chart(view.set_index('Method')[['prompt tokens / query (k)']],horizontal=True)
    else: st.warning('table_cost.csv not found')
    st.markdown("**Benchmark retrieval quality**")
    acc_file = DATA_DIR / "table_accuracy.csv"
    if acc_file.exists():
        raw_acc = pd.read_csv(acc_file)
        st.dataframe(raw_acc,hide_index=True,use_container_width=True)
        quality = None
        st.caption("Accuracy measurements from the saved benchmark table.")
    else:
        quality=None
        st.warning("Accuracy table not available. Add table_accuracy.csv to the q50 directory.")
    st.caption("Benchmark results are loaded from recorded files. One-time summary generation costs are excluded from online inference totals.")
with tabs[2]:
    st.subheader("Scoring and evaluation")
    st.markdown("""
**Periscope:** split each document into short spans; score **local (L)** and **strided global (S)** spans for the question. Relevance = `max(L) + max(S)`.

**SUMScore:** use the same **local (L)** spans, but replace repeatedly computed global probes with a question-specific score for a previously generated **document summary (G)**. Relevance = `max(L) + max(G)`.

A document summary is generated **once offline**, but the summary's relevance is scored again for each question. Short, unsummarized documents reuse their local probe (no extra summary call).

**The cost display** adds the saved prompt-token counts and call counts for each question-document pair. No new tokens or calls are spent by this CPU-only replay.

**Scope:** The dashboard replays recorded retrieval scores; it does not generate answers or evaluate unseen queries. Original question text can be supplied through `queries_<domain>.json`. New inference requires the scoring model.
""")
