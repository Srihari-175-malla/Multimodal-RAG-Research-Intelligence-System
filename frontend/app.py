"""
Streamlit UI for the Multimodal RAG Research Intelligence System.
Talks to the FastAPI backend over HTTP (set API_URL if not running locally).
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Research Intelligence RAG", layout="wide")
st.title("📚 Multimodal RAG Research Intelligence System")

tab_ask, tab_ingest, tab_compare, tab_claim, tab_stats = st.tabs(
    ["Ask", "Ingest Papers", "Compare Papers", "Verify Claim", "Index Stats"]
)

with tab_ask:
    st.subheader("Ask a research question")
    question = st.text_area("Question", placeholder="What methods reduce catastrophic forgetting?")
    top_k = st.slider("Number of sources", 3, 20, 8)
    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving and generating answer..."):
            resp = requests.post(f"{API_URL}/query", json={"question": question, "top_k": top_k})
        if resp.ok:
            data = resp.json()
            st.markdown("### Answer")
            st.markdown(data["answer"])
            st.markdown("### Sources")
            for i, src in enumerate(data["sources"], start=1):
                with st.expander(f"[{i}] {src.get('doc_title','?')} — {src.get('section','?')} (p.{src.get('page','?')})"):
                    st.write(src["text"])
        else:
            st.error(resp.text)

with tab_ingest:
    st.subheader("Upload a research paper (PDF)")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded and st.button("Ingest"):
        with st.spinner("Parsing, chunking, embedding, indexing..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            resp = requests.post(f"{API_URL}/ingest", files=files)
        if resp.ok:
            data = resp.json()
            st.success(f"Ingested '{data['doc_id']}' — {data['num_chunks']} chunks indexed")
            st.json(data["metadata"])
        else:
            st.error(resp.text)

with tab_compare:
    st.subheader("Compare papers on an aspect")
    paper_ids_raw = st.text_input("Paper IDs (comma-separated, e.g. paper1, paper2)")
    aspect = st.text_input("Aspect to compare", placeholder="retrieval strategy")
    if st.button("Compare") and paper_ids_raw and aspect:
        paper_ids = [p.strip() for p in paper_ids_raw.split(",") if p.strip()]
        with st.spinner("Comparing..."):
            resp = requests.post(f"{API_URL}/compare", json={"paper_ids": paper_ids, "aspect": aspect})
        if resp.ok:
            data = resp.json()
            st.markdown(data["answer"])
        else:
            st.error(resp.text)

with tab_claim:
    st.subheader("Find papers supporting or contradicting a claim")
    claim = st.text_area("Claim", placeholder="Larger batch sizes always improve contrastive pretraining.")
    if st.button("Verify") and claim:
        with st.spinner("Searching for evidence..."):
            resp = requests.post(f"{API_URL}/verify_claim", json={"claim": claim})
        if resp.ok:
            data = resp.json()
            st.markdown(data["verdict_explanation"])
            st.markdown("### Evidence excerpts")
            for i, src in enumerate(data["sources"], start=1):
                with st.expander(f"[{i}] {src.get('doc_title','?')} — p.{src.get('page','?')}"):
                    st.write(src["text"])
        else:
            st.error(resp.text)

with tab_stats:
    st.subheader("Index statistics")
    if st.button("Refresh stats"):
        resp = requests.get(f"{API_URL}/stats")
        if resp.ok:
            st.json(resp.json())
        else:
            st.error(resp.text)
