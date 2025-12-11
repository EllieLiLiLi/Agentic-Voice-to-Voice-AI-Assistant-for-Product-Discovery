"""Streamlit UI for the agentic voice-to-voice assistant (chatbot style)."""
from __future__ import annotations

from typing import Any, Dict, List

import io
import os
import streamlit as st
import pandas as pd

from src.asr_tts.asr import transcribe_audio
from src.asr_tts.tts import synthesize_speech


# ==================== Mock Agent Result：先撑 UI ====================

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
                "rating and price; cross-checked some items with web.search."
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


# ==================== 主应用（Chatbot UI） ====================

def app() -> None:
    st.set_page_config(
        page_title="Agentic Voice Product Chat",
        page_icon="🛒",
        layout="wide",
    )

    st.title("🛒 Agentic Voice-to-Voice Product Assistant")

    st.markdown(
        """
This UI currently uses:

- ✅ **Real ASR / TTS** from `src/asr_tts`
- ⚠️ **Mock agent result** (LangGraph & MCP not wired yet)

Once the agent is ready, replace the mock call with the real agent.
"""
    )

    # ---------- 初始化状态 ----------
    if "chat_history" not in st.session_state:
        # 每条消息: {"role": "user"/"assistant", "content": str, "products"?: list, "steps"?: list}
        st.session_state.chat_history: List[Dict[str, Any]] = []

    if "pending_text" not in st.session_state:
        st.session_state.pending_text = ""

    if "last_agent_result" not in st.session_state:
        st.session_state.last_agent_result: Dict[str, Any] | None = None

    if "audio_reply_path" not in st.session_state:
        st.session_state.audio_reply_path: str | None = None

    # ========== 上半部分：聊天记录（chat bubbles） ==========
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # 产品推荐表格（可展开）
                if msg.get("products"):
                    with st.expander("📊 View recommended products"):
                        df = pd.DataFrame(msg["products"])
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
                        st.dataframe(df[cols], use_container_width=True)

                # LangGraph step log（可展开）
                if msg.get("steps"):
                    with st.expander("🪜 View reasoning steps"):
                        for i, step in enumerate(msg["steps"], start=1):
                            node = step.get("node", f"step_{i}")
                            summary = step.get("summary", "")
                            st.markdown(f"**{i}. {node}**")
                            st.write(summary)

    st.divider()

    # ========== 下半部分：新的 query 输入区 ==========
    st.subheader("🎙️ New voice query")

    col_left, col_right = st.columns([1.1, 1.3])

    # ---- 左侧：录音 / 上传 + ASR ----
    with col_left:
        st.markdown("**Record your voice**")
        recorded_audio = st.audio_input("Tap to record")

        st.markdown("—— or — —")

        uploaded_audio = st.file_uploader(
            "Upload audio (WAV / MP3 / M4A)",
            type=["wav", "mp3", "m4a"],
        )

        if st.button("▶️ Run ASR"):
            audio_bytes = None
            filename = "recorded.wav"

            if recorded_audio is not None:
                # st.audio_input 返回的类似 UploadedFile，用 getvalue 拿 bytes
                audio_bytes = recorded_audio.getvalue()
                filename = "recorded.wav"
            elif uploaded_audio is not None:
                audio_bytes = uploaded_audio.read()
                filename = uploaded_audio.name

            if audio_bytes is None:
                st.warning("Please record or upload an audio clip first.")
            else:
                try:
                    transcript = transcribe_audio(audio_bytes, filename=filename)
                    st.session_state.pending_text = transcript
                    st.success("ASR completed.")
                except Exception as e:
                    st.error(f"ASR error: {e}")

    # ---- 右侧：文本编辑 + 发送 & Agent + TTS ----
    with col_right:
        st.markdown("**✍️ Edit or type your query**")
        st.session_state.pending_text = st.text_area(
            "Query text",
            value=st.session_state.pending_text,
            height=100,
        )

        send_col1, send_col2 = st.columns([1, 1])

        # 发送 + 调用 Mock Agent
        with send_col1:
            if st.button("💬 Send & Run Mock Agent", type="primary"):
                user_text = st.session_state.pending_text.strip()
                if not user_text:
                    st.warning("Query is empty.")
                else:
                    # 1) push user 消息
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_text}
                    )

                    # 2) 暂时用 MOCK_AGENT_RESULT，后续换成真实 LangGraph 调用
                    agent_result = MOCK_AGENT_RESULT
                    st.session_state.last_agent_result = agent_result

                    # 3) 生成 assistant 消息
                    assistant_text = agent_result.get("answer", "")
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": assistant_text,
                            "products": agent_result.get("products", []),
                            "steps": agent_result.get("steps", []),
                        }
                    )

                    # 4) 清空输入框
                    st.session_state.pending_text = ""

                    # 5) 立刻 rerun，让上面的聊天记录刷新
                    st.experimental_rerun()

        # TTS：针对最后一次 agent 回复
        with send_col2:
            if st.session_state.last_agent_result is not None:
                if st.button("🔊 TTS for last reply"):
                    try:
                        answer_text = (
                            st.session_state.last_agent_result.get("answer", "") or ""
                        )
                        if not answer_text.strip():
                            st.warning("No answer text to synthesize.")
                        else:
                            audio_bytes_out = synthesize_speech(answer_text)

                            # 写文件再播，最稳
                            out_dir = "tmp_tts"
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, "answer_last.mp3")
                            with open(out_path, "wb") as f:
                                f.write(audio_bytes_out)

                            st.session_state.audio_reply_path = out_path
                            st.success("TTS synthesis completed.")
                    except Exception as e:
                        st.error(f"TTS error: {e}")

        # 播放区域
        if st.session_state.get("audio_reply_path"):
            st.markdown("**▶️ Play synthesized voice**")
            st.audio(st.session_state.audio_reply_path)


if __name__ == "__main__":  # pragma: no cover
    app()
