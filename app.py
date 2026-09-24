"""
app.py
------
Streamlit UI for the AdmissionsAlly RAG chatbot.
Run with:  streamlit run app.py

No extra pip installs needed beyond the base project deps — voice search
uses the browser's own Web Speech API and Urdu translation uses the free
MyMemory API, both called directly from client-side JavaScript.

No sidebar — everything lives in the main/middle area:
  Home    logo, name, description, popular questions, and nav cards
  Chat    the RAG chatbot (a specific session)
  History a list of past chat sessions to reopen
  Universities  a quick-reference list of major Pakistani universities
  Feedback      a simple rating + comment form (saved to feedback.csv)
"""

import os
import csv
import json
import uuid
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from rag_engine import RAGEngine
from agent import Agent

load_dotenv()  # loads GROQ_API_KEY from a local .env file if present

st.set_page_config(page_title="AdmissionsAlly", page_icon="🤝", layout="wide")

FEEDBACK_FILE = "feedback.csv"
HISTORY_FILE = "chat_history.json"
CHAT_WINDOW_HEIGHT = 350  # px — keeps chat and input visible together

POPULAR_QUESTIONS = [
    "What's the eligibility criteria for NUST?",
    "How does the entry test usually work?",
    "What documents do I need for admission?",
    "Are scholarships available for private universities?",
]

# ---- Light, professional academic theme (navy + soft gold on off-white) ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Source+Sans+Pro:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans Pro', sans-serif;
    color: #2B2E33;
}

.stApp {
    background-color: #FAFAF8;
}

h1, h2, h3 {
    font-family: 'Merriweather', serif !important;
    color: #1F3A5F !important;
}

.stButton > button {
    background-color: #FFFFFF;
    border: 1px solid #C7CDD6;
    color: #1F3A5F;
    border-radius: 10px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #1F3A5F;
    color: #FFFFFF;
    border-color: #1F3A5F;
}
.stButton > button[kind="primary"] {
    background-color: #B08D2B;
    border-color: #B08D2B;
    color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover {
    background-color: #1F3A5F;
    border-color: #1F3A5F;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: #6B7280 !important;
}

[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    border: 1px solid #C7CDD6 !important;
}

hr { border-color: #DCE1E8 !important; }

[data-testid="stAlert"] {
    background-color: #F2F4F7;
    border-radius: 8px;
}

/* University reference cards */
.uni-card {
    background-color: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-left: 4px solid #B08D2B;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
}
.uni-card b { color: #1F3A5F; }
.uni-tag {
    display: inline-block;
    font-size: 0.75rem;
    background-color: #F2F4F7;
    color: #1F3A5F;
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
    margin-left: 0.4rem;
    border: 1px solid #DCE1E8;
}

/* ---- Home hero (logo + name + description) ---- */
.aa-hero {
    text-align: center;
    padding: 2.2rem 1rem 1.4rem 1rem;
}
.aa-logo-badge {
    width: 84px;
    height: 84px;
    margin: 0 auto 0.9rem auto;
    border-radius: 50%;
    background: linear-gradient(135deg, #1F3A5F, #2C4F7C);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.4rem;
    box-shadow: 0 4px 14px rgba(31, 58, 95, 0.30);
    border: 3px solid #B08D2B;
}
.aa-hero h1 {
    font-size: 2.1rem;
    margin: 0;
}
.aa-hero p.tagline {
    color: #B08D2B;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0.3rem 0 0.9rem 0;
}
.aa-hero p.desc {
    color: #4B5563;
    font-size: 1rem;
    max-width: 640px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Popular question chips */
.aa-chip-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: #B08D2B;
    margin: 1.4rem 0 0.6rem 0;
    text-align: center;
}

/* Nav cards on home */
.aa-nav-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin: 1.8rem 0 0.6rem 0;
    text-align: center;
}
.aa-nav-grid .stButton > button {
    white-space: pre-line;
    height: 128px;
    line-height: 1.5;
    font-size: 0.95rem;
}

/* History cards */
.aa-hist-card {
    background-color: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-radius: 10px;
    padding: 0.4rem 0.2rem;
    margin-bottom: 0.3rem;
}

/* Banner title inside the navy chat header */
.aa-banner-title, .aa-banner-title * {
    color: #FFFFFF !important;
}
.aa-banner-sub {
    color: #D6E0EC !important;
}
.aa-logo-badge svg, .aa-banner-logo svg {
    display: block;
}

/* Input row: text box + mic + send, all in one line */
.aa-input-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.aa-input-row [data-testid="stTextInput"] { flex: 1; }
.aa-input-row [data-testid="stTextInput"] > div { margin-bottom: 0 !important; }
.aa-input-row iframe { margin-top: 1px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load and Save helper functions for Persistent Chat History
# ---------------------------------------------------------------------------
def load_saved_sessions():
    """Disk se saved chat sessions load karta hai"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_sessions_to_disk(sessions_dict):
    """Chat sessions ko disk (JSON file) par save karta hai"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions_dict, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving chat history: {e}")


# ---------------------------------------------------------------------------
# Logo: a clean SVG monogram badge (navy circle, gold ring, gold "A")
# ---------------------------------------------------------------------------
def logo_svg(size: int = 84) -> str:
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="46" fill="#1F3A5F" stroke="#B08D2B" stroke-width="4"/>
        <circle cx="50" cy="50" r="46" fill="none" stroke="#2C4F7C" stroke-width="1" opacity="0.5"/>
        <text x="50" y="67" font-family="Merriweather, Georgia, serif" font-size="50"
              font-weight="700" fill="#B08D2B" text-anchor="middle">A</text>
    </svg>
    """


# ---------------------------------------------------------------------------
# Reference data: major Pakistani universities (generic, illustrative list)
# ---------------------------------------------------------------------------
UNIVERSITIES = [
    {"name": "Quaid-i-Azam University (QAU)", "city": "Islamabad", "type": "Public",
     "focus": "Sciences, Social Sciences", "url": "https://qau.edu.pk"},
    {"name": "Lahore University of Management Sciences (LUMS)", "city": "Lahore", "type": "Private",
     "focus": "Business, Sciences, Law", "url": "https://lums.edu.pk"},
    {"name": "National University of Sciences & Technology (NUST)", "city": "Islamabad", "type": "Public",
     "focus": "Engineering, Business, Sciences", "url": "https://nust.edu.pk"},
    {"name": "University of the Punjab (PU)", "city": "Lahore", "type": "Public",
     "focus": "Wide range of disciplines", "url": "https://pu.edu.pk"},
    {"name": "Aga Khan University (AKU)", "city": "Karachi", "type": "Private",
     "focus": "Medicine, Nursing", "url": "https://aku.edu"},
    {"name": "Institute of Business Administration (IBA)", "city": "Karachi", "type": "Public",
     "focus": "Business, Computer Science", "url": "https://iba.edu.pk"},
    {"name": "FAST National University (FAST-NU)", "city": "Multiple campuses", "type": "Private",
     "focus": "Computer Science, Engineering, Business", "url": "https://nu.edu.pk"},
    {"name": "Ghulam Ishaq Khan Institute (GIKI)", "city": "Swabi", "type": "Private",
     "focus": "Engineering, Computer Science", "url": "https://giki.edu.pk"},
    {"name": "University of Engineering and Technology (UET)", "city": "Lahore", "type": "Public",
     "focus": "Engineering", "url": "https://uet.edu.pk"},
    {"name": "King Edward Medical University (KEMU)", "city": "Lahore", "type": "Public",
     "focus": "Medicine", "url": "https://kemu.edu.pk"},
    {"name": "Dow University of Health Sciences", "city": "Karachi", "type": "Public",
     "focus": "Medicine, Health Sciences", "url": "https://duhs.edu.pk"},
    {"name": "COMSATS University", "city": "Multiple campuses", "type": "Public",
     "focus": "Sciences, Engineering, Business", "url": "https://comsats.edu.pk"},
]

USER_AVATAR = "🙋"
BOT_AVATAR = "🤝"


def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


@st.cache_resource(show_spinner="Loading knowledge base and models...")
def load_engine():
    api_key = get_api_key()
    rag_engine = RAGEngine(groq_api_key=api_key)
    agent = Agent(groq_api_key=api_key, rag_engine=rag_engine)
    return agent

def new_session():
    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        "title": "New chat",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": [],
    }
    st.session_state.active_session = session_id
    save_sessions_to_disk(st.session_state.sessions)
    return session_id


def init_state():
    if "sessions" not in st.session_state:
        st.session_state.sessions = load_saved_sessions()
    if "active_session" not in st.session_state:
        st.session_state.active_session = None
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "input_generation" not in st.session_state:
        st.session_state.input_generation = 0  # bumped after each send to reset the text box


def go(page: str):
    st.session_state.page = page


def save_feedback(rating: int, comment: str):
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "rating", "comment"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rating, comment])


def back_button():
    if st.button("← Back to Home", key=f"back_{st.session_state.page}"):
        go("Home")
        st.rerun()


def ask_and_store(engine, session, question: str):
    """Run the agent for the question and append both turns to the session."""
    session["messages"].append({"role": "user", "content": question})

    if session["title"] == "New chat":
        session["title"] = question[:40] + ("..." if len(question) > 40 else "")

    history_for_model = [
        {"role": m["role"], "content": m["content"]}
        for m in session["messages"][:-1]
    ]

    try:
        answer, trace = engine.run(
            question,
            chat_history=history_for_model
        )

        # Extract source filenames from the RAG tool result
        sources = []
        for item in trace:
            if item.get("tool") == "retrieve_knowledge_base":
                result = item.get("result", "")
                if "[admissions_guide.txt]" in result:
                    sources.append("admissions_guide.txt")

        sources = list(dict.fromkeys(sources))

    except Exception as e:
        answer = f"Sorry, something went wrong: {e}"
        sources = []

    session["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    
    # Disk par chat history save karna
    save_sessions_to_disk(st.session_state.sessions)


def voice_mic_component(target_label: str, key: str, height: int = 42):
    """Browser-native mic button (Web Speech API) — free, no server round-trip,
    no API key. Finds the real Streamlit text_input by its (hidden) label in
    the parent document, sets its value, and dispatches an 'input' event so
    Streamlit's own state picks up the change. Chrome / Edge / Safari only."""
    components.html(
        f"""
        <div style="display:flex;align-items:center;justify-content:center;height:100%;">
          <button id="aa-mic-{key}" type="button" title="Speak your question (English)"
            style="width:36px;height:36px;border-radius:50%;border:1px solid #C7CDD6;
            background:#FFFFFF;cursor:pointer;font-size:0.95rem;line-height:1;padding:0;
            display:flex;align-items:center;justify-content:center;
            transition:background .15s, color .15s;">🎤</button>
        </div>
        <script>
        (function() {{
            const btn = document.getElementById('aa-mic-{key}');
            const targetLabel = {target_label!r};
            let recognizing = false;

            function getTargetInput() {{
                const doc = window.parent.document;
                const wrappers = doc.querySelectorAll('div[data-testid="stTextInput"]');
                for (const w of wrappers) {{
                    const label = w.querySelector('label');
                    if (label && label.textContent.trim() === targetLabel) {{
                        return w.querySelector('input');
                    }}
                }}
                return null;
            }}

            function setValue(input, text) {{
                const proto = window.parent.HTMLInputElement.prototype;
                const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                setter.call(input, text);
                input.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}

            btn.addEventListener('click', function() {{
                if (recognizing) return;
                const SR = window.webkitSpeechRecognition || window.SpeechRecognition;
                if (!SR) {{
                    alert('Voice search needs Chrome, Edge, or Safari.');
                    return;
                }}
                const recognition = new SR();
                recognition.lang = 'en-US';   // English only, as requested
                recognition.interimResults = false;
                recognition.continuous = false;

                recognition.onstart = function() {{
                    recognizing = true;
                    btn.style.background = '#1F3A5F';
                    btn.style.color = '#FFFFFF';
                }};
                recognition.onresult = function(event) {{
                    const text = event.results[0][0].transcript;
                    const input = getTargetInput();
                    if (input) setValue(input, text);
                }};
                recognition.onerror = function() {{
                    recognizing = false;
                    btn.style.background = '#FFFFFF';
                    btn.style.color = '#000000';
                }};
                recognition.onend = function() {{
                    recognizing = false;
                    btn.style.background = '#FFFFFF';
                    btn.style.color = '#000000';
                }};
                recognition.start();
            }});
        }})();
        </script>
        """,
        height=height,
    )


# ---------------------------------------------------------------------------
# HOME: logo -> name -> description -> popular questions -> nav cards
# ---------------------------------------------------------------------------
def render_home(engine):
    st.markdown(
        f"""
        <div class="aa-hero">
            <div class="aa-logo-badge">{logo_svg(88)}</div>
            <h1>AdmissionsAlly</h1>
            <p class="tagline">Your Admissions Companion</p>
            <p class="desc">Ask about eligibility criteria, entry tests, deadlines, documents,
            or scholarships — general guidance for Pakistan university admissions. Always confirm
            exact details on your target university's official website.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="aa-chip-label">✨ Popular questions</p>', unsafe_allow_html=True)
    chip_cols = st.columns(len(POPULAR_QUESTIONS))
    for col, q in zip(chip_cols, POPULAR_QUESTIONS):
        with col:
            if st.button(q, key=f"chip_{q}", use_container_width=True):
                sid = new_session()
                with st.spinner("Thinking..."):
                    ask_and_store(engine, st.session_state.sessions[sid], q)
                go("Chat")
                st.rerun()

    st.markdown('<p class="aa-nav-label">Explore</p>', unsafe_allow_html=True)
    st.markdown('<div class="aa-nav-grid">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💬\n\nNew Chat\nStart a fresh conversation", use_container_width=True):
            new_session()
            go("Chat")
            st.rerun()
    with c2:
        if st.button("📊\n\nEligibility Calculator\nEstimate your merit score", use_container_width=True):
            go("Eligibility")
            st.rerun()
    with c3:
        if st.button("🏫\n\nUniversities\nBrowse major universities", use_container_width=True):
            go("Universities")
            st.rerun()

    c4, c5, _ = st.columns(3)
    with c4:
        if st.button("📝\n\nFeedback\nShare suggestions", use_container_width=True):
            go("Feedback")
            st.rerun()
    with c5:
        n = len(st.session_state.sessions)
        if st.button(f"🕘\n\nChat History\n{n} saved chat(s)", use_container_width=True):
            go("History")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HISTORY: past sessions as clickable cards
# ---------------------------------------------------------------------------
def render_history_page():
    back_button()
    st.header("🕘 Chat History")

    past_sessions = sorted(
        st.session_state.sessions.items(),
        key=lambda kv: kv[1]["created"],
        reverse=True,
    )
    if not past_sessions:
        st.info("No chats yet — start one from the Home screen.")
        return

    for session_id, session in past_sessions:
        title = session["title"] or "New chat"
        with st.container(border=True):
            col_text, col_btn = st.columns([4, 1])
            with col_text:
                st.markdown(f"**{title}**")
                st.caption(f"{session['created']} · {len(session['messages'])} messages")
            with col_btn:
                if st.button("Open →", key=f"open_{session_id}", use_container_width=True):
                    st.session_state.active_session = session_id
                    go("Chat")
                    st.rerun()


# ---------------------------------------------------------------------------
# CHAT — the whole conversation is rendered as ONE HTML/JS component:
#   - fixed-height, internally-scrollable panel (like the Rahber chat-view)
#   - the Urdu button lives INSIDE each bot bubble
#   - clicking it calls MyMemory's translate API straight from the browser
#     and swaps that bubble's own content in place — no new bubble, no
#     Streamlit rerun, exactly like Rahber's toggleBotTranslation().
# ---------------------------------------------------------------------------
def render_chat_window(session):
    messages = session["messages"]
    msgs_payload = json.dumps([
        {
            "role": m["role"],
            "content": m["content"],
            "sources": m.get("sources") or [],
        }
        for m in messages
    ])

    html = f"""
    <style>
    #aa-chat-window {{
        height: {CHAT_WINDOW_HEIGHT}px;
        overflow-y: auto;
        padding: 0.8rem;
        font-family: 'Source Sans Pro', Arial, sans-serif;
        box-sizing: border-box;
        border: 1px solid #E4E7EC;
        border-radius: 14px;
        background: #F8F9FB;
    }}
    .aa-empty {{
        height: 100%; display:flex; flex-direction:column;
        align-items:center; justify-content:center; text-align:center;
        color:#7A8799;
    }}
    .aa-empty-icon {{
        width:74px; height:74px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        background:linear-gradient(135deg,#1F3A5F,#C9A227);
        box-shadow:0 8px 20px rgba(16,24,40,.12);
        margin-bottom:14px; font-size:36px;
    }}
    .aa-empty-title {{
        color:#1F3A5F; font-size:1.15rem; font-weight:700;
        margin-bottom:7px;
    }}
    .aa-empty-text {{
        max-width:560px; font-size:.9rem; line-height:1.55;
    }}
    .aa-empty-hint {{
        margin-top:9px; font-size:.8rem; color:#98A2B3;
    }}
    .aa-row {{ display:flex; width:100%; align-items:flex-end; gap:0.5rem; margin-bottom:0.5rem; }}
    .aa-row.user {{ justify-content:flex-end; }}
    .aa-row.bot  {{ justify-content:flex-start; }}
    .aa-avatar {{
        width:30px; height:30px; border-radius:50%; display:flex; align-items:center;
        justify-content:center; font-size:1.05rem; flex-shrink:0; background:#F2F4F7;
        border:1px solid #E4E7EC;
    }}
    .aa-bubble {{
        max-width:74%; padding:0.55rem 0.85rem; border-radius:14px; line-height:1.45;
        font-size:0.95rem; box-shadow:0 1px 2px rgba(16,24,40,0.06); word-wrap:break-word;
    }}
    .aa-bubble.user {{
        background:#DCF3D1; border:1px solid #BFE6AF; border-bottom-right-radius:3px; color:#1B2A1F;
    }}
    .aa-bubble.bot {{
        background:#FFFFFF; border:1px solid #E4E7EC; border-bottom-left-radius:3px; color:#2B2E33;
    }}
    .aa-sources {{ margin-top:0.35rem; font-size:0.76rem; color:#6B7280; }}
    .aa-translate-row {{ margin-top:0.45rem; padding-top:0.4rem; border-top:1px dashed #E4E7EC; }}
    .aa-translate-btn {{
        border:1px solid #C7CDD6; background:#F8F9FB; color:#1F3A5F; border-radius:999px;
        padding:0.18rem 0.7rem; font-size:0.76rem; cursor:pointer; font-weight:600;
    }}
    .aa-translate-btn:hover {{ background:#1F3A5F; color:#FFFFFF; border-color:#1F3A5F; }}
    .aa-content[data-lang="ur"] {{
        direction:rtl; text-align:right; font-family:'Noto Nastaliq Urdu','Jameel Noori Nastaleeq',serif;
        font-size:1.05rem; color:#1F3A5F;
    }}
    </style>
    <div id="aa-chat-window"><div id="aa-chat-inner"></div></div>
    <script>
    (function() {{
        const messages = {msgs_payload};
        const container = document.getElementById('aa-chat-inner');

        function escapeHtml(s) {{
            return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }}

        function englishHtml(msg) {{
            let html = escapeHtml(msg.content).replace(/\\n/g, '<br/>');
            if (msg.sources && msg.sources.length) {{
                html += '<div class="aa-sources">📚 Sources: ' + msg.sources.map(escapeHtml).join(', ') + '</div>';
            }}
            return html;
        }}

        function bubbleHtml(msg, idx) {{
            const isUser = msg.role === 'user';
            const rowCls = isUser ? 'user' : 'bot';
            const avatar = '<div class="aa-avatar">' + (isUser ? '🙋' : '🤝') + '</div>';
            let translateRow = '';
            if (!isUser) {{
                translateRow = '<div class="aa-translate-row">' +
                    '<button type="button" class="aa-translate-btn" id="aa-btn-' + idx + '" ' +
                    'onclick="aaToggleTranslate(' + idx + ')">🌐 Urdu Translate</button></div>';
            }}
            const content = '<div class="aa-content" id="aa-content-' + idx + '" data-lang="en">' +
                englishHtml(msg) + '</div>' + translateRow;
            const bubble = '<div class="aa-bubble ' + rowCls + '">' + content + '</div>';
            const inner = isUser ? (bubble + avatar) : (avatar + bubble);
            return '<div class="aa-row ' + rowCls + '">' + inner + '</div>';
        }}

        let out = '';
        if (!messages.length) {{
            out = '<div class="aa-empty">' +
                  '<div class="aa-empty-icon">🎓</div>' +
                  '<div class="aa-empty-title">How can I help with your admission?</div>' +
                  '<div class="aa-empty-text">Ask me about university eligibility, entry tests, required documents, admission deadlines, scholarships, or programs.</div>' +
                  '<div class="aa-empty-hint">Try: &quot;What is the eligibility criteria for NUST?&quot;</div>' +
                  '</div>';
        }} else {{
            messages.forEach(function(m, idx) {{ out += bubbleHtml(m, idx); }});
        }}
        container.innerHTML = out;

        const win = document.getElementById('aa-chat-window');
        win.scrollTop = win.scrollHeight;

        window.aaToggleTranslate = async function(idx) {{
            const contentDiv = document.getElementById('aa-content-' + idx);
            const btn = document.getElementById('aa-btn-' + idx);
            const lang = contentDiv.getAttribute('data-lang');
            const msg = messages[idx];

            if (lang === 'ur') {{
                contentDiv.innerHTML = englishHtml(msg);
                contentDiv.setAttribute('data-lang', 'en');
                btn.innerHTML = '🌐 Urdu Translate';
                return;
            }}

            const original = btn.innerHTML;
            btn.innerHTML = '⏳ Translating...';
            try {{
                const chunks = original ? (msg.content.match(/[\\s\\S]{{1,450}}/g) || [msg.content]) : [msg.content];
                const parts = [];
                for (const c of chunks) {{
                    const res = await fetch('https://api.mymemory.translated.net/get?q=' +
                        encodeURIComponent(c) + '&langpair=en|ur');
                    const data = await res.json();
                    parts.push((data.responseData && data.responseData.translatedText) || c);
                }}
                contentDiv.innerHTML = parts.join(' ');
                contentDiv.setAttribute('data-lang', 'ur');
                btn.innerHTML = '↩️ Show English';
            }} catch (e) {{
                btn.innerHTML = '⚠️ Failed — retry';
                setTimeout(function() {{ btn.innerHTML = '🌐 Urdu Translate'; }}, 2000);
            }}
        }};
    }})();
    </script>
    """
    components.html(html, height=CHAT_WINDOW_HEIGHT + 10, scrolling=False)


def render_chat_page(engine):
    back_button()

    if st.session_state.active_session is None:
        new_session()
    session_id = st.session_state.active_session
    session = st.session_state.sessions[session_id]

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1F3A5F, #2C4F7C);
                    padding: 1.1rem 1.5rem; border-radius: 14px; margin: 0.75rem 0 1.1rem 0;
                    display: flex; align-items: center; gap: 0.8rem;">
            <div class="aa-banner-logo">{logo_svg(40)}</div>
            <h2 class="aa-banner-title" style="margin:0; font-size:1.35rem;">AdmissionsAlly</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_chat_window(session)

    gen = st.session_state.input_generation
    input_key = f"chat_text_input_{session_id}_{gen}"

    with st.form(key=f"chat_form_{session_id}_{gen}", clear_on_submit=True, border=False):
        st.markdown('<div class="aa-input-row">', unsafe_allow_html=True)
        col_text, col_mic, col_send = st.columns([10, 0.8, 1.4])

        with col_text:
            user_input = st.text_input(
                input_key,
                key=input_key,
                label_visibility="collapsed",
                placeholder="Ask about eligibility, entry tests, deadlines, documents...",
            )

        with col_mic:
            voice_mic_component(
                target_label=input_key,
                key=f"mic_{session_id}_{gen}"
            )

        with col_send:
            send_clicked = st.form_submit_button(
                "➤ Send",
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    submitted_text = user_input.strip() if user_input else ""

    if st.session_state.pending_question and not submitted_text:
        submitted_text = st.session_state.pending_question

    st.session_state.pending_question = None

    if send_clicked and submitted_text:
        with st.spinner("Thinking..."):
            ask_and_store(engine, session, submitted_text)

        st.session_state.input_generation += 1
        st.rerun()


def render_eligibility_page():
    back_button()
    st.header("📊 Eligibility / Merit Calculator")
    st.caption(
        "A generic weighted-merit estimate (matric % × 10, intermediate % × 40, entry test % × 50). "
        "Actual weightages differ by university — confirm on the official admissions page."
    )

    with st.form("eligibility_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            matric = st.number_input("Matric %", min_value=0.0, max_value=100.0, value=80.0, step=0.5)
        with col2:
            inter = st.number_input("Intermediate %", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
        with col3:
            test = st.number_input("Entry test %", min_value=0.0, max_value=100.0, value=70.0, step=0.5)

        with st.expander("⚙️ Advanced: custom weightages"):
            w1, w2, w3 = st.columns(3)
            with w1:
                weight_matric = st.number_input("Matric weight", min_value=0.0, max_value=1.0, value=0.10, step=0.05)
            with w2:
                weight_inter = st.number_input("Inter weight", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
            with w3:
                weight_test = st.number_input("Test weight", min_value=0.0, max_value=1.0, value=0.50, step=0.05)

        submitted = st.form_submit_button("Calculate merit", type="primary")

    if submitted:
        merit = (matric * weight_matric) + (inter * weight_inter) + (test * weight_test)
        st.success(f"### Estimated merit score: {merit:.2f}%")
        st.caption(
            f"Formula used: matric×{weight_matric} + inter×{weight_inter} + test×{weight_test}. "
            "This is a general estimate — some universities weigh subjects differently or add "
            "interview/portfolio components."
        )


def render_universities_page():
    back_button()
    st.header("🏫 University Directory")
    st.caption(
        "A quick-reference list of well-known Pakistani universities. This is illustrative, not "
        "exhaustive — always check each university's official site for current programs and deadlines."
    )

    search = st.text_input("🔍 Search by name, city, or focus area")
    filtered = UNIVERSITIES
    if search:
        s = search.lower()
        filtered = [
            u for u in UNIVERSITIES
            if s in u["name"].lower() or s in u["city"].lower() or s in u["focus"].lower()
        ]

    if not filtered:
        st.info("No universities match your search.")
        return

    for u in filtered:
        st.markdown(
            f"""
            <div class="uni-card">
                <b>{u['name']}</b><span class="uni-tag">{u['type']}</span>
                <br/>📍 {u['city']} &nbsp;|&nbsp; 🎯 {u['focus']}
                <br/><a href="{u['url']}" target="_blank">{u['url']}</a>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feedback_page():
    back_button()
    st.header("📝 Feedback & Suggestions")
    st.caption(
        "Found this helpful? Something missing or wrong? Let us know — your feedback helps improve "
        "this guide."
    )

    with st.form("feedback_form", clear_on_submit=True):
        rating = st.radio(
            "How helpful was this chatbot?",
            [1, 2, 3, 4, 5],
            index=4,
            horizontal=True,
            format_func=lambda x: "⭐" * x,
        )
        comment = st.text_area(
            "Comments or suggestions (missing info, incorrect details, feature ideas, etc.)"
        )
        submitted = st.form_submit_button("Submit feedback", type="primary")

    if submitted:
        save_feedback(rating, comment)
        st.success("Thank you for your feedback!")


def main():
    init_state()
    engine = load_engine()

    page = st.session_state.page
    if page == "Home":
        render_home(engine)
    elif page == "Chat":
        render_chat_page(engine)
    elif page == "History":
        render_history_page()
    elif page == "Eligibility":
        render_eligibility_page()
    elif page == "Universities":
        render_universities_page()
    elif page == "Feedback":
        render_feedback_page()


if __name__ == "__main__":
    main()