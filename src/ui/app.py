"""Streamlit UI for the agentic voice-to-voice assistant."""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ✅ 用你们自己实现的 ASR / TTS 函数（已经在 src/asr_tts 里）
from src.asr_tts.asr import transcribe_audio
from src.asr_tts.tts import synthesize_speech


# =========================
# 1. 假的 Agent 返回结果（先用来撑 UI）
# =========================
MOCK_AGENT_RESULT: Dict[str, Any] = {
    "answer": (
        "Here are some eco-friendly stainless-steel cleaners under $15 that "
        "match your request. I prioritized high rating and plant-based ingredients."
    ),
    "steps": [
        {
            "node": "router",
            "summary": (
                "Detected intent as product recommendation for stainless-steel "
                "cleaner with eco-friendly and price < $15 constraints."
            ),
        },
        {
            "node": "planner",
            "summary": (
                "Planned to call rag.search over the Amazon 2020 cleaning slice, "
                "filtering by category='cleaning', max_price=15, and eco-friendly features."
            ),
        },
        {
            "node": "retriever",
            "summary": (
                "Retrieved top 5 items from local vector index and re-ranked them by "
                "rating and price; cross-checked a couple of items with web.search."
            ),
        },
        {
            "node": "answerer",
            "summary": (
                "Generated a concise explanation, highlighted 2–3 best options, and "
                "attached citations (local doc_id + live URLs)."
            ),
        },
    ],
    "products": [
        {
            "sku": "B00ABC123",
            "title": "Eco Stainless Steel Cleaner Spray, 16oz",
            "brand": "GreenHome",
            "price": 12.99,
            "rating": 4.6,
            "doc_id": "local-123",
            "source_url": "https://example.com/product/eco-steel-1",
        },
        {
            "sku": "B00XYZ456",
            "title": "Plant-Based Stainless Steel Wipes, 50 count",
            "brand": "CleanLeaf",
            "price": 9.49,
            "rating": 4.4,
            "doc_id": "local-456",
            "source_url": "https://example.com/product/eco-steel-2",
        },
        {
            "sku": "B00QWE789",
            "title": "Fragrance-Free Steel Cleaner with Refill Pack",
            "brand": "PureShine",
            "price": 13.50,
            "rating": 4.5,
            "doc_id": "local-789",
            "source_url": "https://example.com/product/eco-steel-3",
        },
    ],
}


def render_audio_recorder() -> Optional[bytes]:
    """Streamlit 自带的录音控件，返回音频 bytes。"""
    audio_bytes = st.audio_input("🎤 Record a voice query", key="mic")
    return audio_bytes


def app() -> None:
    # ========== 页面配置 ==========
    st.set_page_config(
        page_title="Voice Product Assistant",
        page_icon="🛒",
        layout="wide",
    )

    st.title("🛒 Agentic Voice-to-Voice Product Discovery")

    st.markdown(
        """
This UI currently uses:

- ✅ **Real ASR / TTS** from `src/asr_tts`
- ⚠️ **Mock agent result** (LangGraph & MCP not wired yet)

Once the agent is ready, replace the mock with real calls.
"""
    )

    # ========== 初始化 session_state ==========
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None
    if "audio_reply" not in st.session_state:
        st.session_state.audio_reply = None

    # ========== 左右两栏 ==========
    left_col, right_col = st.columns([1.1, 1.3])

    # ==============================
    # 左：录音 + ASR + Transcript + Mock Agent + TTS
    # ==============================
    with left_col:
        st.subheader("🎙️ Voice Input & Controls")

        # 1) 录音控件
        audio_bytes = render_audio_recorder()

        # 2) 跑 ASR（用你们的 transcribe_audio）
        if st.button("▶️ Run ASR"):
            if not audio_bytes:
                st.warning("Please record or upload audio first.")
            else:
                try:
                    transcript = transcribe_audio(audio_bytes, filename="query.wav")
                    st.session_state.transcript = transcript
                    st.success("ASR completed.")
                except Exception as e:
                    st.error(f"ASR error: {e}")

        # 3) Transcript 文本框（可手动改）
        st.markdown("### ✍️ Transcript (editable)")
        st.session_state.transcript = st.text_area(
            "You can edit or type your query here:",
            value=st.session_state.transcript,
            height=150,
        )

        # 4) 跑 Mock Agent（先把右边 UI 的长相撑出来）
        if st.button("🤖 Run Mock Agent (fake LangGraph)"):
            if not st.session_state.transcript.strip():
                st.warning("Transcript is empty. Type something first.")
            else:
                st.session_state.agent_result = MOCK_AGENT_RESULT
                st.success("Mock agent result loaded.")

        # 5) 用 TTS 播放回答（用你们的 synthesize_speech）
        agent_result = st.session_state.agent_result
        if agent_result and agent_result.get("answer"):
            st.markdown("### 🔊 TTS (play answer)")
            if st.button("Generate & Play Voice Reply"):
                try:
                    audio_bytes_out = synthesize_speech(agent_result["answer"])
                    st.session_state.audio_reply = audio_bytes_out
                except Exception as e:
                    st.error(f"TTS error: {e}")

            if st.session_state.audio_reply:
                st.audio(st.session_state.audio_reply, format="audio/mp3")

    # ==============================
    # 右：Agent reasoning + Product table + Citations
    # ==============================
    with right_col:
        st.subheader("🧠 Agent Reasoning & Product Results")

        agent_result = st.session_state.agent_result

        if agent_result is None:
            st.info("Click **Run Mock Agent** on the left to see example output.")
            return

        # 1) Final Answer
        st.markdown("### ✅ Final Answer")
        st.write(agent_result.get("answer", ""))

        # 2) Step Log（LangGraph trace 的样子）
        st.markdown("### 🪜 Agent Step Log (mock)")
        steps: List[Dict[str, Any]] = agent_result.get("steps", [])
        if not steps:
            st.write("No step log provided.")
        else:
            for i, step in enumerate(steps, start=1):
                node_name = step.get("node", f"step_{i}")
                summary = step.get("summary", "")
                with st.expander(f"{i}. {node_name}"):
                    st.write(summary)

        # 3) 产品对比表
        st.markdown("### 📊 Top-K Product Comparison (mock)")
        products: List[Dict[str, Any]] = agent_result.get("products", [])
        if not products:
            st.write("No products returned.")
        else:
            df = pd.DataFrame(products)
            preferred_cols = [
                "sku",
                "title",
                "brand",
                "price",
                "rating",
                "doc_id",
                "source_url",
            ]
            cols = [c for c in preferred_cols if c in df.columns] + [
                c for c in df.columns if c not in preferred_cols
            ]
            df = df[cols]
            st.dataframe(df, use_container_width=True)

        # 4) 引用信息（doc_id + URL）
        st.markdown("### 🔗 Citations")
        if not products:
            st.write("No citations.")
        else:
            for p in products:
                doc_id = p.get("doc_id")
                url = p.get("source_url")
                title = p.get("title") or p.get("sku")
                if doc_id or url:
                    line = []
                    if doc_id:
                        line.append(f"**doc_id:** `{doc_id}`")
                    if url:
                        line.append(f"[{title}]({url})")
                    st.markdown("- " + " — ".join(line))


if __name__ == "__main__":  # pragma: no cover
    app()
