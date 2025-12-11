"""Streamlit UI for the agentic voice-to-voice assistant (chatbot style)."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.asr_tts.asr import transcribe_audio
from src.asr_tts.tts import synthesize_speech


# =========================
# 1. Mock Agent output (replace with real LangGraph call later)
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
            "node": "critic",
            "summary": (
                "Ensured that recommended items are actually stainless-steel cleaners, "
                "not general-purpose or abrasive products, and that they are in stock."
            ),
        },
        {
            "node": "final_answer",
            "summary": (
                "Summarized pros/cons for each cleaner and recommended two best options "
                "with short justifications and price/rating info."
            ),
        },
    ],
    "products": [
        {
            "sku": "B00ABC123",
            "title": "Eco-Friendly Stainless Steel Cleaner Spray, 500ml",
            "brand": "GreenShine",
            "price": 12.99,
            "rating": 4.7,
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


def run_agent(query: str) -> Dict[str, Any]:
    """Placeholder for your real LangGraph / RAG pipeline."""
    result = dict(MOCK_AGENT_RESULT)
    result["question"] = query
    return result


# =========================
# 2. Helper: synthesize TTS for each answer
# =========================
def synthesize_answer_audio(answer_text: str) -> str | None:
    """Generate TTS for the answer and return the audio file path."""
    if not answer_text:
        return None

    try:
        audio_bytes_out = synthesize_speech(answer_text)
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

    out_dir = "tmp_tts"
    os.makedirs(out_dir, exist_ok=True)

    # Use current message count to create a unique filename
    msg_index = len(st.session_state.get("messages", []))
    out_path = os.path.join(out_dir, f"reply_{msg_index}.mp3")
    with open(out_path, "wb") as f:
        f.write(audio_bytes_out)

    return out_path


# =========================
# 3. Render details (no expander here!)
# =========================
def render_agent_details(agent_result: Dict[str, Any]) -> None:
    """Render step log, product table and citations (no outer expander)."""
    steps: List[Dict[str, Any]] = agent_result.get("steps", [])
    products: List[Dict[str, Any]] = agent_result.get("products", [])

    # # Step log
    # st.markdown("#### Agent Step Log")
    # if not steps:
    #     st.write("No step log provided.")
    # else:
    #     for i, step in enumerate(steps, start=1):
    #         node_name = step.get("node", f"step_{i}")
    #         summary = step.get("summary", "")
    #         st.markdown(f"**{i}. {node_name}**")
    #         st.write(summary)
    #         st.markdown("---")

    # Product comparison table
    st.markdown("#### Top-3 Product Comparison")
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

    # Citations
    st.markdown("#### Citations")
    if not products:
        st.write("No citations.")
    else:
        for p in products:
            doc_id = p.get("doc_id")
            url = p.get("source_url")
            title = p.get("title") or p.get("sku")
            if not (doc_id or url):
                continue
            line_parts = []
            if doc_id:
                line_parts.append(f"**doc_id:** `{doc_id}`")
            if url:
                line_parts.append(f"[{title}]({url})")
            st.markdown("- " + " — ".join(line_parts))


# =========================
# 4. Main app
# =========================
def app() -> None:
    st.set_page_config(
        page_title="Agentic Voice-to-Voice Product Assistant",
        page_icon="🛒",
        layout="wide",
    )

    # background color
    st.markdown(
        """
        <style>
        /* ===== Layout backgrounds ===== */
    
        /* Left sidebar: solid teal */
        [data-testid="stSidebar"] {
            background-color: #88ada5;
        }
    
        /* Main app view (right side): very light greenish */
        [data-testid="stAppViewContainer"] {
            background-color: #f5faf4;
        }
    
        /* Top app bar: slightly darker than main background */
        header[data-testid="stHeader"] {
            background-color: #e4efe4 !important;
            box-shadow: none !important;
        }
        header[data-testid="stHeader"] > div {
            background-color: #e4efe4 !important;
        }

        /* ===== Bottom chat input bar ===== */
    
        /* Bar container: keep top line */
        [data-testid="stChatInput"] {
            background-color: #ffffff;
            border-top: 1px solid #d0ddd4;   /* keep this line */
            padding: 0.9rem 1.5rem 1.2rem 1.5rem;
        }
    
        /* Outer pill around the chat input */
        [data-testid="stChatInput"] > div {
            border-radius: 999px !important;
            border: 1px solid #b7cbbf !important;  /* only one visible border */
            box-shadow: none !important;
            background-color: #f6f8fb !important;
        }
    
        /* Text area: no own border, so it aligns with outer pill */
        [data-testid="stChatInput"] textarea {
            border: none !important;
            background-color: transparent !important;
            box-shadow: none !important;
            outline: none !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }
    
        /* Focus state: subtle teal glow on the outer pill */
        [data-testid="stChatInput"] textarea:focus-visible {
            outline: none !important;
        }
        [data-testid="stChatInput"] > div:has(textarea:focus) {
            border-color: #88ada5 !important;
            box-shadow: 0 0 0 1px #88ada533;
        }
    
        /* ===== Sidebar boxes ===== */
    
        /* Audio recorder card (top box) -> white */
        [data-testid="stSidebar"] [data-testid="stAudioInput"] > div {
            background-color: #ffffff !important;
            border-radius: 16px;
        }
    
        /* File uploader dropzone (second box) -> white with solid border */
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background-color: #ffffff !important;
            border-radius: 16px;
            border: 1px solid #d0ddd4;
        }
    
        /* "Browse files" button inside dropzone -> deeper teal */
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            background-color: #88ada5 !important;
            color: #ffffff !important;
            border-radius: 999px;
            border: none;
            box-shadow: 0 1px 2px rgba(0,0,0,0.16);
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
            background-color: #76958f !important;
        }
    
        /* Sidebar buttons (Send voice / Clear conversation)
           -> same color as right background */
        [data-testid="stSidebar"] .stButton > button {
            background-color: #f5faf4 !important;
            color: #24332c !important;
            border-radius: 999px;
            border: none;
            box-shadow: 0 1px 2px rgba(0,0,0,0.12);
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #e5f0e6 !important;
        }
    
        /* ===== Chat bubbles ===== */
    
        /* Remove any default background from chat message wrapper */
        [data-testid="stChatMessage"] {
            background-color: transparent;
        }
    
        /* Inner content of each chat message as rounded bubble */
        [data-testid="stChatMessage"] > div {
            border-radius: 16px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        }
    
        /* Heuristic: odd = user, even = assistant.
           If it looks reversed, swap these two blocks. */
    
        /* User messages: white bubble */
        [data-testid="stChatMessage"]:nth-of-type(odd) > div {
            background-color: #ffffff;
        }
    
        /* Assistant messages: soft green bubble */
        [data-testid="stChatMessage"]:nth-of-type(even) > div {
            background-color: #d9ead3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <h2 style="
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.4rem;
        ">
          Agentic Voice-to-Voice Product Discovery Assistant
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
Ask me via text or voice. I will provide a voice response along with optional text, product tables, and citations.
"""
    )

    # ----- Session state -----
    if "messages" not in st.session_state:
        # Each message: {role, content, agent_result?, audio_path?}
        st.session_state.messages: List[Dict[str, Any]] = []

    # ----- Sidebar: voice input -----
    with st.sidebar:
        st.header("🎙️ Voice input")
        recorded_audio = st.audio_input("Record your question")
        st.markdown("—— or ——")
        audio_file = st.file_uploader(
            "Upload a short voice query (WAV / MP3 / M4A)",
            type=["wav", "mp3", "m4a"],
        )

        if st.button("Send voice to chatbot"):
            audio_bytes = None
            filename = "recorded.wav"

            if recorded_audio is not None:
                audio_bytes = recorded_audio.getvalue()
                filename = "recorded.wav"
            elif audio_file is not None:
                audio_bytes = audio_file.read()
                filename = audio_file.name

            if audio_bytes is None:
                st.warning("Please record or upload an audio clip first.")
            else:
                try:
                    transcript = transcribe_audio(audio_bytes, filename=filename)
                except Exception as e:
                    st.error(f"ASR error: {e}")
                else:
                    # 1) Add user message
                    st.session_state.messages.append(
                        {"role": "user", "content": transcript}
                    )

                    # 2) Run agent
                    agent_result = run_agent(transcript)
                    answer_text = agent_result.get("answer", "")

                    # 3) Auto-generate TTS
                    audio_path = synthesize_answer_audio(answer_text)

                    # 4) Add assistant message with audio
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "agent_result": agent_result,
                            "audio_path": audio_path,
                        }
                    )
                    st.success("Voice query sent to chatbot.")
                    st.rerun()

        st.markdown("---")
        if st.button("🧹 Clear conversation"):
            # Optional: try to clean up audio files
            for msg in st.session_state.messages:
                ap = msg.get("audio_path")
                if ap and os.path.exists(ap):
                    try:
                        os.remove(ap)
                    except OSError:
                        pass
            st.session_state.messages = []
            st.rerun()

    # ----- Main area: chat history -----
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        with st.chat_message("user" if role == "user" else "assistant"):
            if role == "user":
                st.markdown(msg.get("content", ""))
            else:
                # 1) Voice answer
                audio_path = msg.get("audio_path")
                if audio_path and os.path.exists(audio_path):
                    st.audio(audio_path)
                else:
                    st.write("No audio available for this answer.")

                # 2) Dropdown with text + products + citations
                with st.expander("View full answer and product information"):
                    st.markdown("#### Answer")
                    st.markdown(msg.get("content", ""))
                    agent_result = msg.get("agent_result")
                    if agent_result:
                        render_agent_details(agent_result)

    # ----- Chat input (text) -----
    user_text = st.chat_input("Type your product question here…")
    if user_text:
        # 1) Add user message
        st.session_state.messages.append({"role": "user", "content": user_text})

        # 2) Run agent
        agent_result = run_agent(user_text)
        answer_text = agent_result.get("answer", "")

        # 3) Auto-generate TTS
        audio_path = synthesize_answer_audio(answer_text)

        # 4) Add assistant message with audio
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "agent_result": agent_result,
                "audio_path": audio_path,
            }
        )
        st.rerun()


if __name__ == "__main__":  # pragma: no cover
    app()
