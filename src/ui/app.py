"""Streamlit UI for the agentic voice-to-voice assistant (chatbot style)."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.asr_tts.asr import transcribe_audio
from src.asr_tts.tts import synthesize_speech
from src.graph.graph import agent as product_agent
from src.graph.nodes import (
    router_node,
    planner_node,
    retriever_node,
    answerer_node,
)

# =========================
# =========================
# 1'. Agent runner（LangGraph backend）
# =========================
def run_agent(query: str) -> Dict[str, Any]:
    """Run the toy product assistant by calling the 4 LangGraph nodes in order.

    Router → Planner → Retriever → Answerer
    """


    state: Dict[str, Any] = {
        "user_query": query,
        "intent": {},
        "constraints": {},
        "plan": [],
        "search_strategy": None,
        "search_params": {},
        "rag_results": [],
        "web_results": [],
        "reconciled_results": [],
        "final_answer": {},
        "citations": [],
        "node_logs": [],
    }

    try:
        # 1) Router
        state = router_node(state)

        # if router returns out_of_scope，skip
        if state.get("intent", {}).get("type") != "out_of_scope":
            # 2) Planner
            state = planner_node(state)
            # 3) Retriever
            state = retriever_node(state)
            # 4) Answerer
            state = answerer_node(state)

    except Exception as e:
  
        st.error(f"Agent error: {e}")
        return {
            "answer": f"[Agent error] {e}",
            "products": [],
            "steps": [],
            "raw_state": {"error": repr(e)},
        }


    final_answer: Dict[str, Any] = state.get("final_answer", {}) or {}
    spoken = final_answer.get("spoken_summary")
    detailed = final_answer.get("detailed_analysis")
    answer_text = (
        spoken
        or detailed
        or "I generated a result, but could not read the final answer."
    )

    products = (
        state.get("reconciled_results")
        or state.get("rag_results")
        or state.get("web_results")
        or []
    )

    logs = state.get("node_logs") or []
    steps = [
        {"node": f"step_{i+1}", "summary": log}
        for i, log in enumerate(logs)
    ]

    return {
        "answer": answer_text,
        "products": products,
        "steps": steps,
        "raw_state": state,
    }



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

    def _extract_image_url(prod: Dict[str, Any]) -> str | None:
        """Best-effort lookup for a product image/thumbnail URL."""

        if not prod:
            return None

        def _coerce_url(val: Any) -> str | None:
            if isinstance(val, str):
                if val.startswith("http"):
                    return val
                if val.startswith("//"):
                    return "https:" + val
            return None

        # Common single-value fields
        for key in (
            "image",
            "image_url",
            "imageUrl",
            "image_link",
            "imageLink",
            "thumbnail",
            "thumbnail_url",
            "thumbnailUrl",
            "main_image",
            "mainImage",
        ):
            url = _coerce_url(prod.get(key))
            if url:
                return url

        # Handle dict wrappers like {"url": ...} or {"src": ...}
        for key in ("image", "thumbnail", "primaryImage"):
            val = prod.get(key)
            if isinstance(val, dict):
                url = _coerce_url(val.get("url") or val.get("src"))
                if url:
                    return url

        # Handle lists of images
        for key in ("images", "image_urls", "imageUrls", "thumbnails"):
            val = prod.get(key)
            if isinstance(val, list):
                for candidate in val:
                    if isinstance(candidate, dict):
                        url = _coerce_url(candidate.get("url") or candidate.get("src"))
                    else:
                        url = _coerce_url(candidate)
                    if url:
                        return url

        # Search any field with "image" in the key as a last resort
        for key, val in prod.items():
            if "image" in key.lower() or "thumb" in key.lower():
                url = _coerce_url(val)
                if not url and isinstance(val, dict):
                    url = _coerce_url(val.get("url") or val.get("src"))
                if not url and isinstance(val, list):
                    for candidate in val:
                        url = _coerce_url(candidate.get("url") if isinstance(candidate, dict) else candidate)
                        if url:
                            break
                if url:
                    return url

        return None

    raw_state = agent_result.get("raw_state", {}) or {}

    base_citations: List[Dict[str, Any]] = raw_state.get("citations", []) or agent_result.get(
        "citations", []
    )

    # Fallback: build citations from the returned products when the LLM omitted them
    fallback_citations: List[Dict[str, Any]] = []
    if not base_citations and products:
        seen_keys = set()
        for prod in products:
            url = prod.get("url")
            product_id = prod.get("product_id") or prod.get("sku") or prod.get("id")
            key = url or product_id or prod.get("title")
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            fallback_citations.append(
                {
                    "type": prod.get("source") or "web",
                    "id": product_id,
                    "url": url,
                    "title": prod.get("title") or "(untitled product)",
                    "price": prod.get("price"),
                    "rating": prod.get("rating"),
                }
            )

    citations = base_citations or fallback_citations

    citation_index = {}
    for idx, cit in enumerate(citations, start=1):
        for key in [cit.get("url"), cit.get("id"), cit.get("title")]:
            if key and key not in citation_index:
                citation_index[key] = idx

    # ===== 0) Agent Step Log =====
    st.markdown("#### Agent Step Log")
    if not steps:
        st.write("No step log provided.")
    else:
        for i, step in enumerate(steps, start=1):
            node_name = step.get("node", f"step_{i}")
            summary = step.get("summary", "")
            st.markdown(f"**{i}. {node_name}**")
            st.write(summary)
            st.markdown("---")

        # ===== 1) Product Comparison =====
    st.markdown("#### Top Product Comparison")
    if not products:
        st.write("No products returned.")
    else:
        st.markdown("##### Product photos")
        with st.container(border=True):
            top_cards = products[:3]
            cols = st.columns(len(top_cards)) if top_cards else []
            for col, product in zip(cols, top_cards):
                with col:
                    img_url = _extract_image_url(product)
                    if img_url:
                        st.image(img_url, use_column_width=True)
                    else:
                        st.markdown(
                            "<div style='height:180px; display:flex; align-items:center; justify-content:center; background:#f0f4f2; border-radius:12px; color:#6b7a70;'>No image</div>",
                            unsafe_allow_html=True,
                        )

                    citation_id = (
                        citation_index.get(product.get("url"))
                        or citation_index.get(product.get("product_id"))
                        or citation_index.get(product.get("sku"))
                        or citation_index.get(product.get("id"))
                        or citation_index.get(product.get("title"))
                    )

                    title = product.get("title", "Untitled product")
                    if citation_id:
                        title = f"{title} [{citation_id}]"

                    st.markdown(f"**{title}**")
                    price = product.get("price")
                    rating = product.get("rating")
                    details = []
                    if price is not None:
                        details.append(f"${price}")
                    if rating is not None:
                        details.append(f"⭐ {rating}")
                    brand = product.get("brand")
                    if brand:
                        details.append(str(brand))
                    if details:
                        st.caption(" • ".join(map(str, details)))
                    url = product.get("url")
                    if url:
                        st.markdown(f"[View product]({url})")

        df = pd.DataFrame(products)

        df = df.drop(columns=["score"], errors="ignore")

        preferred_cols = [
            "sku",
            "title",
            "brand",
            "price",
            "rating",
            "url",
            "source",
        ]
        cols = [c for c in preferred_cols if c in df.columns] + [
            c for c in df.columns if c not in preferred_cols
        ]
        df = df[cols]

        st.dataframe(df, use_container_width=True)

 
    st.markdown("#### Citations")
    if not citations:
        st.write("No citations.")
    else:
        for i, c in enumerate(citations, start=1):
            title = c.get("title") or "(no title)"
            url = c.get("url")

            if not url:
                cid = c.get("id")
                url = cid if isinstance(cid, str) and cid.startswith("http") else None

            suffix_parts = []
            if c.get("type"):
                suffix_parts.append(c["type"])
            if c.get("price") is not None:
                suffix_parts.append(f"${c['price']}")
            if c.get("rating") is not None:
                suffix_parts.append(f"⭐ {c['rating']}")
            suffix = f" — {' • '.join(map(str, suffix_parts))}" if suffix_parts else ""

            if url:
                st.markdown(f"{i}. [{title}]({url}){suffix}")
            else:
                st.markdown(f"{i}. {title}{suffix}")




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
        <p style="
            font-size: 0.95rem;
            margin-bottom: 0.9rem;
            color: #37474f;
        ">
          Ask about products by voice or text. The assistant will search our product catalog
          and the web, then respond with a concise spoken answer plus detailed product info.
        </p>
        """,
        unsafe_allow_html=True,
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
                    raw_answer = msg.get("content", "")
                
                    safe_answer = raw_answer.replace("$", r"\$")
                
                    st.markdown(safe_answer)
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
