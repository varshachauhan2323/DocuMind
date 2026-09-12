import hashlib
import json
import os
import re
import threading
import uuid
from datetime import date

import streamlit as st

# =====================================================
# PAGE CONFIG (must be the very first Streamlit call)
# =====================================================
st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# DEFENSIVE IMPORTS
# If any of these fail (e.g. version mismatch), show a
# clear error in the browser instead of a blank page.
# =====================================================
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_community.vectorstores import FAISS
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.runnables import RunnableLambda
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    try:
        # LangChain >= 1.0 moved these legacy chain helpers here
        from langchain_classic.chains import create_history_aware_retriever
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    except ImportError:
        # Older LangChain versions
        from langchain.chains import create_history_aware_retriever
        from langchain.chains.combine_documents import create_stuff_documents_chain

    IMPORTS_OK = True
    IMPORT_ERROR = None
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = e

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

try:
    # CookieController is the one piece of streamlit-authenticator we still
    # need: an encrypted browser cookie so a Supabase session can survive
    # closing/reopening the app on the same device. Credential checking
    # (Authenticate/Hasher) is no longer used — Supabase Auth now owns
    # signup/login/password hashing/verification.
    from streamlit_authenticator import CookieController
    AUTH_IMPORT_ERROR = None
except ImportError as error:
    CookieController = None
    AUTH_IMPORT_ERROR = error

try:
    from supabase import create_client
    SUPABASE_IMPORT_ERROR = None
except ImportError as error:
    create_client = None
    SUPABASE_IMPORT_ERROR = error

# =====================================================
# CUSTOM CSS
# Scoped to decorative elements only. `.stApp` gets an
# explicit stacking context (isolation: isolate) so the
# fixed, negative-z-index glow blobs can never paint over
# real widgets in any browser. Text-color rules are scoped
# to plain text containers only — never to generic
# containers that Streamlit reuses for buttons, inputs,
# etc., which is what can make a widget look "invisible"
# if its own text color is later overridden.
# =====================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(160deg, #f8fafc 0%, #eef2ff 45%, #f5f3ff 100%);
        overflow-x: hidden;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        height: 60px;
        background: #ffffff;
        border-bottom: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 4px 18px rgba(15,23,42,0.06);
    }
    header[data-testid="stHeader"]::before {
        content: "📚  DocuMind  |  PDF Intelligence Assistant";
        white-space: pre;
        display: block;
        color: #0f172a;
        font-size: 0.98rem;
        font-weight: 750;
        line-height: 1;
        padding: 1.1rem 1.05rem;
    }
    header[data-testid="stHeader"]::after {
        content: "RAG Powered";
        position: absolute;
        right: 8rem;
        top: 1.1rem;
        color: #2563eb;
        font-size: 0.62rem;
        font-weight: 650;
        letter-spacing: 0.02em;
        padding: 0.3rem 0.48rem;
        border: 1px solid rgba(37,99,235,0.18);
        border-radius: 999px;
        background: rgba(37,99,235,0.05);
    }
    .stMarkdown, .stCaption, section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: #1e293b;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff, #f1f5f9);
        border-right: 1px solid rgba(148,163,184,0.25);
    }
    .hero-container {
        padding: 1.4rem 1.6rem 1.6rem;
        background: rgba(255,255,255,0.55);
        border-radius: 24px;
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 22px 48px rgba(15,23,42,0.10), 0 4px 14px rgba(15,23,42,0.05);
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-size: 2.65rem; font-weight: 800; color: #0f172a;
        line-height: 1.12; letter-spacing: -1px; margin-bottom: 0.45rem;
        text-shadow: 0 2px 10px rgba(37,99,235,0.10);
    }
    .hero-highlight { color: #2563eb; }
    .hero-subtitle { font-size: 1.08rem; color: #334155; max-width: 760px; line-height: 1.5; margin-bottom: 0.35rem; }
    .hero-description { font-size: 0.98rem; color: #64748b; max-width: 820px; line-height: 1.55; margin-bottom: 0.55rem; }
    .hero-capabilities { color: #64748b; font-size: 0.76rem; letter-spacing: 0.04em; }
    .welcome-card {
        background: rgba(255,255,255,0.72); backdrop-filter: blur(14px);
        border: 1px solid rgba(148,163,184,0.25); border-radius: 22px;
        padding: 32px 28px; text-align: center; margin-top: 12px;
        box-shadow: 0 20px 45px rgba(15,23,42,0.08);
    }
    .welcome-icon { font-size: 2.5rem; margin-bottom: 8px; }
    .welcome-title { color: #0f172a; font-size: 2rem; font-weight: 700; margin-bottom: 15px; }
    .welcome-text { color: #475569; font-size: 1.05rem; line-height: 1.7; max-width: 650px; margin: auto; }
    .info-card {
        background: rgba(255,255,255,0.68); backdrop-filter: blur(12px);
        border: 1px solid rgba(148,163,184,0.22); border-radius: 18px;
        padding: 21px; min-height: 165px;
        box-shadow: 0 10px 30px rgba(15,23,42,0.06);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-5px); border-color: #60a5fa;
        box-shadow: 0 16px 36px rgba(37,99,235,0.12);
    }
    .info-icon { color: #2563eb; height: 24px; margin-bottom: 12px; }
    .info-icon svg { width: 24px; height: 24px; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
    .info-title { color: #0f172a; font-size: 1.2rem; font-weight: 700; margin-bottom: 8px; }
    .info-text { color: #475569; line-height: 1.6; font-size: 0.95rem; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.72); border: 1px solid rgba(37,99,235,0.15);
        border-radius: 15px; padding: 18px; box-shadow: 0 8px 24px rgba(15,23,42,0.05);
    }
    .stButton button {
        border-radius: 10px; border: 1px solid rgba(37,99,235,0.35);
        background: rgba(255,255,255,0.85); color: #1e293b; transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px); border-color: #2563eb;
        box-shadow: 0 8px 20px rgba(37,99,235,0.15);
    }
    [data-testid="stChatInput"] {
        border-radius: 16px; background: rgba(255,255,255,0.85);
        border: 1px solid rgba(148,163,184,0.25);
    }
    .made-by { text-align: center; color: #64748b; font-size: 0.85rem; padding: 10px 0; }
    .made-by b { color: #2563eb; }
    .sidebar-user-card {
        border: 1px solid rgba(148,163,184,0.28);
        border-radius: 12px;
        padding: 12px;
        background: rgba(255,255,255,0.55);
        margin-top: 4px;
    }
    .sidebar-user-label { color: #64748b; font-size: 0.78rem; margin-bottom: 2px; }
    .sidebar-user-name { color: #0f172a; font-size: 0.92rem; font-weight: 650; }
    .sidebar-brand { color: #0f172a; font-size: 1.18rem; font-weight: 800; line-height: 1.2; }
    .sidebar-descriptor { color: #64748b; font-size: 0.76rem; margin-top: 2px; }
    @media (max-width: 640px) {
        header[data-testid="stHeader"]::before { font-size: 0.92rem; padding-left: 0.65rem; }
        header[data-testid="stHeader"]::after { right: 2.9rem; font-size: 0.56rem; padding: 0.28rem 0.4rem; }
    }
    hr { border-color: rgba(148,163,184,0.25); }

    /* =====================================================
       MOBILE RESPONSIVENESS (320px–768px)
       Generic, structural rules only — no branding/color
       changes, no touching desktop layout above 768px.
       ===================================================== */

    /* Never let anything force horizontal scroll of the page. */
    html, body, .stApp { max-width: 100vw; overflow-x: hidden; }

    @media (max-width: 768px) {
        /* Every st.columns(...) row (auth card ratio, quick-action
           buttons, info cards, compare-mode selectors) stacks to a
           single column instead of squeezing into narrow slivers. */
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 0 !important;
        }

        .hero-container { padding: 1rem 1.1rem 1.2rem; border-radius: 16px; }
        .hero-title { font-size: 1.65rem; letter-spacing: -0.5px; }
        .hero-subtitle { font-size: 0.95rem; }
        .hero-description { font-size: 0.88rem; }
        .hero-capabilities { font-size: 0.68rem; }

        .welcome-card { padding: 22px 16px; border-radius: 16px; }
        .welcome-title { font-size: 1.4rem; }
        .welcome-text { font-size: 0.95rem; }

        .info-card { min-height: 0; padding: 16px; }
        .info-title { font-size: 1.05rem; }
        .info-text { font-size: 0.9rem; }

        /* Touch targets: buttons and inputs comfortably tappable. */
        .stButton button, div[data-testid="stForm"] button {
            min-height: 44px;
            font-size: 0.95rem;
            width: 100%;
        }
        div[data-testid="stForm"] { padding: 1rem 1rem 1.1rem; }
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {
            min-height: 44px;
            font-size: 0.95rem;
        }

        /* Chat messages, citations, and any wide block (tables,
           retrieval-detail dumps) scroll within themselves instead of
           blowing out the page width. */
        [data-testid="stChatMessage"] { max-width: 100%; }
        [data-testid="stChatMessageContent"] { word-break: break-word; }
        .stMarkdown table, div[data-testid="stExpander"] {
            display: block;
            max-width: 100%;
            overflow-x: auto;
        }
        [data-testid="stChatInput"] textarea { font-size: 0.95rem; }

        /* Sidebar takes the full width on a phone instead of a
           fixed desktop-sized panel. */
        section[data-testid="stSidebar"] { min-width: 0 !important; width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_NAME = "openai/gpt-oss-20b"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RETRIEVER_K = 10
RETRIEVER_FETCH_K = 30
FAISS_K = 10
BM25_K = 10
RERANK_K = 5
EVIDENCE_MAX_CHARS = 600
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# =====================================================
# PER-USER RATE LIMITING
#
# Each authenticated user (identified by their login email —
# the same value used for local login and Google OAuth) gets
# their own daily question allowance. Usage is tracked in a
# small JSON file on disk (not st.session_state) so it:
#   - survives page refreshes / reruns,
#   - survives the Streamlit server restarting,
#   - is never shared between different users' counts.
#
# This is deliberately separate from Groq/provider-side outages
# (see is_provider_quota_error below) — a shared provider limit
# should never be reported to a user as "you personally used up
# your 20 questions".
# =====================================================
USER_DAILY_QUESTION_LIMIT = int(st.secrets.get("USER_DAILY_QUESTION_LIMIT", 20))
USAGE_STORE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".documind_usage_store.json"
)
_usage_lock = threading.Lock()


def _today_str() -> str:
    return date.today().isoformat()


def _load_usage_store() -> dict:
    try:
        with open(USAGE_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_usage_store(store: dict) -> None:
    tmp_path = f"{USAGE_STORE_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(store, f)
    os.replace(tmp_path, USAGE_STORE_PATH)


def get_user_usage_count(user_key: str) -> int:
    """Today's question count for this user. Resets automatically on a new day."""
    if not user_key:
        return 0
    with _usage_lock:
        record = _load_usage_store().get(user_key)
    if not record or record.get("date") != _today_str():
        return 0
    return int(record.get("count", 0))


def user_has_remaining_quota(user_key: str) -> bool:
    return get_user_usage_count(user_key) < USER_DAILY_QUESTION_LIMIT


def increment_user_usage(user_key: str) -> int:
    """Atomically records one question against this user's daily allowance.

    Keyed strictly by user_key, so User A's questions never touch User B's count.
    """
    if not user_key:
        return 0
    with _usage_lock:
        store = _load_usage_store()
        today = _today_str()
        record = store.get(user_key)
        if not record or record.get("date") != today:
            record = {"date": today, "count": 0}
        record["count"] += 1
        store[user_key] = record
        _write_usage_store(store)
        return record["count"]


def refund_user_usage(user_key: str) -> None:
    """Give back one question when a request failed for reasons that aren't the
    user's fault (e.g. the shared Groq/provider quota was exhausted), so a
    provider-side outage never eats into someone's personal daily allowance."""
    if not user_key:
        return
    with _usage_lock:
        store = _load_usage_store()
        today = _today_str()
        record = store.get(user_key)
        if record and record.get("date") == today and record.get("count", 0) > 0:
            record["count"] -= 1
            store[user_key] = record
            _write_usage_store(store)


def is_provider_quota_error(exc: Exception) -> bool:
    """True when a Groq call failed because of a shared/provider-side limit
    (rate limit or quota exhaustion) rather than a normal application error.

    This is intentionally heuristic: it checks the common attributes the groq
    SDK's exceptions expose (status_code / body) as well as the exception
    text, since the exact exception class can vary by langchain-groq version.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    text = " ".join(
        str(part) for part in (
            exc.__class__.__name__,
            str(exc),
            getattr(exc, "body", ""),
            getattr(exc, "message", ""),
        )
        if part
    ).lower()

    return any(
        marker in text
        for marker in (
            "rate_limit", "rate limit", "ratelimit",
            "quota", "insufficient_quota",
            "429", "too many requests",
            "capacity", "overloaded",
        )
    )


# =====================================================
# SESSION STATE (safe defaults — never accessed before init)
# =====================================================
st.session_state.setdefault("store", {})
st.session_state.setdefault("messages_by_session", {})
st.session_state.setdefault("session_id", uuid.uuid4().hex)
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("pdf_info", {"files": 0, "pages": 0, "chunks": 0})
st.session_state.setdefault("processed_files_fingerprint", None)
st.session_state.setdefault("history_aware_retriever", None)
st.session_state.setdefault("question_answer_chain", None)
st.session_state.setdefault("selected_document", "All Documents")
st.session_state.setdefault("uploaded_files", [])
st.session_state.setdefault("mode", "Chat")
st.session_state.setdefault("compare_document_a", "")
st.session_state.setdefault("compare_document_b", "")
st.session_state.setdefault("document_analytics", {})
st.session_state.setdefault("retrieval_state", {})
st.session_state.setdefault("auth_view", "login")
st.session_state.setdefault("user_app_states", {})
st.session_state.setdefault("active_app_username", None)
st.session_state.setdefault("auth_method", None)
# Supabase session tokens for the CURRENT browser tab only — not a store of
# accounts. The account record itself lives in Supabase, not here.
st.session_state.setdefault("supabase_session", None)
st.session_state.setdefault("password_reset_error", None)
st.session_state.setdefault("upload_just_received", False)

USER_STATE_KEYS = (
    "store",
    "messages_by_session",
    "session_id",
    "pending_question",
    "pdf_info",
    "processed_files_fingerprint",
    "history_aware_retriever",
    "question_answer_chain",
    "selected_document",
    "uploaded_files",
    "mode",
    "compare_document_a",
    "compare_document_b",
    "document_analytics",
    "retrieval_state",
)


def save_current_user_state():
    username = st.session_state.get("active_app_username")
    if username:
        st.session_state.user_app_states[username] = {
            key: st.session_state.get(key) for key in USER_STATE_KEYS
        }


def clear_active_user_state():
    for key in USER_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state.active_app_username = None


def ensure_app_state_defaults():
    st.session_state.setdefault("store", {})
    st.session_state.setdefault("messages_by_session", {})
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("pdf_info", {"files": 0, "pages": 0, "chunks": 0})
    st.session_state.setdefault("processed_files_fingerprint", None)
    st.session_state.setdefault("history_aware_retriever", None)
    st.session_state.setdefault("question_answer_chain", None)
    st.session_state.setdefault("selected_document", "All Documents")
    st.session_state.setdefault("uploaded_files", [])
    st.session_state.setdefault("mode", "Chat")
    st.session_state.setdefault("compare_document_a", "")
    st.session_state.setdefault("compare_document_b", "")
    st.session_state.setdefault("document_analytics", {})
    st.session_state.setdefault("retrieval_state", {})


def activate_user_state(username):
    if st.session_state.get("active_app_username") == username:
        return

    save_current_user_state()
    user_state = st.session_state.user_app_states.get(username)
    if user_state is None:
        user_state = {
            "store": {},
            "messages_by_session": {},
            "session_id": f"user::{username}",
            "pending_question": None,
            "pdf_info": {"files": 0, "pages": 0, "chunks": 0},
            "processed_files_fingerprint": None,
            "history_aware_retriever": None,
            "question_answer_chain": None,
            "selected_document": "All Documents",
            "uploaded_files": [],
            "mode": "Chat",
            "compare_document_a": "",
            "compare_document_b": "",
            "document_analytics": {},
            "retrieval_state": {},
        }
        st.session_state.user_app_states[username] = user_state

    for key in USER_STATE_KEYS:
        st.session_state[key] = user_state[key]
    st.session_state.active_app_username = username


def google_user_value(name, default=""):
    try:
        value = getattr(st.user, name, default)
        if value:
            return value
    except Exception:
        pass
    try:
        return st.user.get(name, default)
    except Exception:
        return default


def activate_google_user():
    """Resolve the Google-authenticated identity (from Streamlit's own OIDC
    session, st.user) to a PERSISTENT Supabase user id, so a Google login and
    an email/password login with the same address land on one account.

    Requires a `profiles` table in Supabase (id uuid references auth.users,
    email text, full_name text) — see the setup notes for the SQL. Uses the
    service-role client (server-side only secret, never sent to the browser)
    because looking a user up by email and creating one when missing needs
    admin privileges that the anon key intentionally doesn't have.
    """
    email = str(google_user_value("email")).strip().lower()
    if not email:
        st.error("Google did not provide an email address. Please try again.")
        return False

    name = google_user_value("name", email)
    admin = get_supabase_admin_client()
    if admin is None:
        st.error(
            "Google Sign-In needs SUPABASE_SERVICE_ROLE_KEY configured to "
            "link accounts. Add it to .streamlit/secrets.toml."
        )
        return False

    try:
        existing = (
            admin.table("profiles")
            .select("id, full_name")
            .eq("email", email)
            .limit(1)
            .execute()
        )
        if existing.data:
            user_id = existing.data[0]["id"]
            display_name = existing.data[0].get("full_name") or name
        else:
            created = admin.auth.admin.create_user({
                "email": email,
                "email_confirm": True,
                "password": uuid.uuid4().hex,
                "user_metadata": {"full_name": name, "auth_provider": "google"},
            })
            user_id = created.user.id
            display_name = name
            # The `profiles` row is normally populated by a DB trigger on
            # auth.users insert (see setup notes); upsert here too so login
            # still works even before that trigger is created.
            admin.table("profiles").upsert({
                "id": user_id, "email": email, "full_name": name,
            }).execute()
    except Exception as e:
        st.error("❌ Could not link your Google account to DocuMind.")
        st.exception(e)
        return False

    st.session_state.current_username = email
    st.session_state.current_user_id = user_id
    st.session_state.current_user = display_name
    st.session_state.auth_method = "google"
    activate_user_state(st.session_state.current_user_id)
    return True


def get_session_history(session: str) -> "BaseChatMessageHistory":
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


def get_conversation_key(
    session_id: str,
    mode: str,
    selected_document: str,
    document_a: str = "",
    document_b: str = "",
) -> str:
    if mode == "Compare":
        return f"{session_id}::Compare::{document_a}::{document_b}"
    return f"{session_id}::Chat::{selected_document}"


def hash_file(uploaded_file) -> str:
    """Content hash so a same-named/same-sized-but-edited PDF is detected as changed."""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def normalize_answer(answer: str) -> str:
    """Convert HTML line-break tags from model output into plain text breaks."""
    return re.sub(r"<br\s*/?>", "\n", answer, flags=re.IGNORECASE)


def get_supabase_client():
    """Anon-key client: safe for signup/login/password-reset — this is what
    an untrusted browser session is allowed to do.

    Deliberately NOT @st.cache_resource: sign_in/sign_up mutate this
    client's internal session state, and a cached instance is shared by
    every user hitting this Streamlit server process — caching it would let
    one user's login leak into another's request. A plain client is cheap
    to construct (no model/network init), so building one per call is fine.
    """
    if create_client is None:
        st.error("Authentication requires the `supabase` package.")
        st.code("pip install supabase")
        if SUPABASE_IMPORT_ERROR is not None:
            st.exception(SUPABASE_IMPORT_ERROR)
        st.stop()
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
    except Exception:
        st.error(
            "Add SUPABASE_URL and SUPABASE_ANON_KEY to "
            ".streamlit/secrets.toml."
        )
        st.stop()
    return create_client(url, key)


@st.cache_resource
def get_supabase_admin_client():
    """Service-role client: only used server-side (Streamlit secrets never
    reach the browser) for the Google-account-linking lookup in
    activate_google_user(). Returns None if not configured, so local
    email/password auth keeps working without it."""
    if create_client is None:
        return None
    url = st.secrets.get("SUPABASE_URL")
    service_key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not service_key:
        return None
    return create_client(url, service_key)


def get_cookie_controller():
    if AUTH_IMPORT_ERROR is not None:
        st.error(f"Authentication import failed: {AUTH_IMPORT_ERROR!r}")
        st.code("pip install streamlit-authenticator")
        st.stop()

    cookie_key = st.secrets.get("AUTH_COOKIE_KEY")
    if not cookie_key:
        groq_key = st.secrets.get("GROQ_API_KEY")
        if not groq_key:
            st.error("Add AUTH_COOKIE_KEY to .streamlit/secrets.toml.")
            st.stop()
        cookie_key = hashlib.sha256(groq_key.encode()).hexdigest()

    return CookieController(cookie_name="documind_supabase_session", key=cookie_key)


def restore_supabase_session(cookie_controller):
    """Reopening the app / a new tab on the SAME device: the account was
    never stored locally, so what's restored here is only the SESSION
    (refresh token) from an encrypted cookie — not the account. Supabase is
    still asked to confirm/refresh it, so a revoked or expired session
    correctly falls back to the login screen."""
    if st.session_state.get("current_user_id"):
        return True

    refresh_token = None
    try:
        refresh_token = cookie_controller.get("documind_supabase_session")
    except Exception:
        pass
    if not refresh_token:
        return False

    supabase = get_supabase_client()
    try:
        result = supabase.auth.refresh_session(refresh_token)
    except Exception:
        cookie_controller.delete_cookie()
        return False

    if not result or not result.session or not result.user:
        cookie_controller.delete_cookie()
        return False

    cookie_controller.set(
        "documind_supabase_session",
        result.session.refresh_token,
        max_age=30 * 24 * 60 * 60,
    )
    _apply_authenticated_supabase_user(result.user, result.session)
    return True


def _apply_authenticated_supabase_user(user, session):
    full_name = (user.user_metadata or {}).get("full_name") or user.email
    st.session_state.current_username = user.email
    st.session_state.current_user_id = user.id
    st.session_state.current_user = full_name
    st.session_state.auth_method = "local"
    st.session_state.supabase_session = {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
    }
    activate_user_state(st.session_state.current_user_id)


def render_authentication(cookie_controller):
    # Supabase redirects back with a recovery link like
    # ?type=recovery&access_token=...&refresh_token=... — catch that first,
    # regardless of whatever auth_view was showing before the link was
    # clicked, and route straight to "set a new password".
    query_params = st.query_params
    if query_params.get("type") == "recovery" and query_params.get("access_token"):
        st.session_state.auth_view = "reset"
        st.session_state.reset_tokens = {
            "access_token": query_params.get("access_token"),
            "refresh_token": query_params.get("refresh_token", ""),
        }
        st.query_params.clear()

    # If the user is already authenticated (Google session persisted by
    # Streamlit, or a valid local Supabase session restored from cookie),
    # skip drawing the entire login/signup screen — including the DocuMind
    # icon/title card — and just report success immediately.
    if getattr(st.user, "is_logged_in", False):
        return activate_google_user()
    if st.session_state.get("current_user_id") and st.session_state.auth_method == "local":
        return True
    if st.session_state.auth_view != "reset" and restore_supabase_session(cookie_controller):
        return True

    st.markdown(
        """
        <style>
            .auth-heading {
                text-align: center;
                color: #0f172a;
                margin: 0.25rem 0 0.2rem;
                font-size: 1.85rem;
                font-weight: 750;
            }
            .auth-brand {
                text-align: center;
                color: #0f172a;
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
            }
            .auth-descriptor {
                text-align: center;
                color: #64748b;
                font-size: 0.85rem;
                margin: 0.25rem 0 1.1rem;
            }
            .auth-subheading {
                text-align: center;
                color: #64748b;
                margin: 0 0 1.25rem;
                font-size: 0.98rem;
            }
            .auth-icon {
                text-align: center;
                font-size: 2rem;
                margin-bottom: 0.15rem;
            }
            div[data-testid="stForm"] {
                border: 1px solid rgba(148,163,184,0.32);
                border-radius: 16px;
                padding: 1.25rem 1.35rem 1.35rem;
                background: rgba(255,255,255,0.76);
                box-shadow: 0 16px 36px rgba(15,23,42,0.08);
            }
            div[data-testid="stForm"] input {
                border-radius: 9px;
                background: #ffffff;
            }
            div[data-testid="stForm"] button,
            div[data-testid="stButton"] button {
                min-height: 2.65rem;
                border-radius: 9px;
                font-weight: 600;
            }
            .auth-divider {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                color: #94a3b8;
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                margin: 1.1rem 0;
            }
            .auth-divider::before,
            .auth-divider::after {
                content: "";
                height: 1px;
                flex: 1;
                background: rgba(148,163,184,0.35);
            }
            .auth-footnote {
                text-align: center;
                color: #64748b;
                margin-top: 1rem;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, card, right = st.columns([1, 1.15, 1])
    with card:
        st.markdown('<div class="auth-icon">📚</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-brand">DocuMind</div>'
            '<div class="auth-descriptor">PDF Intelligence Assistant</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.auth_view == "signup":
        with card:
            st.markdown('<div class="auth-heading">Create your account</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="auth-subheading">Join DocuMind</div>',
                unsafe_allow_html=True,
            )
            with st.form("signup_form"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                create_account = st.form_submit_button("Create Account", use_container_width=True)

        if create_account:
            normalized_email = email.strip().lower()
            if not name.strip() or not normalized_email or not password:
                st.error("Please complete all fields.")
            elif "@" not in normalized_email or "." not in normalized_email.split("@")[-1]:
                st.error("Please enter a valid email address.")
            elif len(password) < 8:
                st.error("Password must contain at least 8 characters.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                supabase = get_supabase_client()
                try:
                    result = supabase.auth.sign_up({
                        "email": normalized_email,
                        "password": password,
                        "options": {"data": {"full_name": name.strip()}},
                    })
                except Exception as e:
                    text = str(e).lower()
                    if "already" in text or "registered" in text or "exists" in text:
                        st.error(
                            "An account with this email already exists. "
                            "Please log in or use a different email."
                        )
                    else:
                        st.error("❌ Could not create your account.")
                        st.exception(e)
                    result = None

                if result is not None:
                    # Best-effort profile row (a DB trigger should normally
                    # do this on auth.users insert — see setup notes — this
                    # is just a safety net if that trigger isn't set up yet).
                    if result.user is not None:
                        try:
                            supabase.table("profiles").upsert({
                                "id": result.user.id,
                                "email": normalized_email,
                                "full_name": name.strip(),
                            }).execute()
                        except Exception:
                            pass

                    st.session_state.auth_view = "login"
                    if result.session is not None:
                        st.success("Account created! Signing you in…")
                    else:
                        st.success(
                            "Account created! Check your email to confirm it, "
                            "then sign in."
                        )
                    st.rerun()

        with card:
            st.markdown(
                '<div class="auth-footnote">Already have an account?</div>',
                unsafe_allow_html=True,
            )
            if st.button("Sign In", use_container_width=True, key="back_to_login"):
                st.session_state.auth_view = "login"
                st.rerun()
        return False

    if st.session_state.auth_view == "forgot":
        with card:
            st.markdown('<div class="auth-heading">Reset your password</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="auth-subheading">We\'ll email you a reset link</div>',
                unsafe_allow_html=True,
            )
            with st.form("forgot_form"):
                reset_email = st.text_input("Email")
                send_reset = st.form_submit_button("Send Reset Link", use_container_width=True)

        if send_reset:
            normalized_email = reset_email.strip().lower()
            if not normalized_email:
                st.error("Please enter your email.")
            else:
                supabase = get_supabase_client()
                try:
                    redirect_to = st.secrets.get("APP_URL", "")
                    supabase.auth.reset_password_for_email(
                        normalized_email,
                        {"redirect_to": redirect_to} if redirect_to else None,
                    )
                except Exception:
                    pass  # Never reveal whether an email is registered.
                st.success(
                    "If an account exists for that email, a reset link is "
                    "on its way."
                )

        with card:
            if st.button("Back to Sign In", use_container_width=True, key="forgot_back"):
                st.session_state.auth_view = "login"
                st.rerun()
        return False

    if st.session_state.auth_view == "reset":
        with card:
            st.markdown('<div class="auth-heading">Set a new password</div>', unsafe_allow_html=True)
            tokens = st.session_state.get("reset_tokens") or {}
            if not tokens.get("access_token"):
                st.error("This reset link is invalid or has expired.")
                if st.button("Back to Sign In", use_container_width=True, key="reset_invalid_back"):
                    st.session_state.auth_view = "login"
                    st.rerun()
                return False

            with st.form("reset_form"):
                new_password = st.text_input("New Password", type="password")
                confirm_new_password = st.text_input("Confirm New Password", type="password")
                submit_reset = st.form_submit_button("Update Password", use_container_width=True)

        if submit_reset:
            if len(new_password) < 8:
                st.error("Password must contain at least 8 characters.")
            elif new_password != confirm_new_password:
                st.error("Passwords do not match.")
            else:
                supabase = get_supabase_client()
                try:
                    supabase.auth.set_session(
                        tokens["access_token"], tokens.get("refresh_token", "")
                    )
                    supabase.auth.update_user({"password": new_password})
                    st.session_state.pop("reset_tokens", None)
                    st.session_state.auth_view = "login"
                    st.success("Password updated. Please sign in.")
                    st.rerun()
                except Exception as e:
                    st.error("❌ Could not update your password. Request a new reset link.")
                    st.exception(e)
        return False

    with card:
        st.markdown('<div class="auth-heading">Welcome back 👋</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-subheading">Sign in to continue</div>',
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            login_email = st.text_input("Email")
            login_password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Sign In", use_container_width=True)
        forgot_clicked = st.button(
            "Forgot Password?", key="open_forgot", use_container_width=True
        )

    if forgot_clicked:
        st.session_state.auth_view = "forgot"
        st.rerun()

    if submit_login:
        normalized_email = login_email.strip().lower()
        supabase = get_supabase_client()
        try:
            result = supabase.auth.sign_in_with_password({
                "email": normalized_email,
                "password": login_password,
            })
        except Exception:
            result = None

        if result is None or result.user is None or result.session is None:
            st.error("Incorrect email or password.")
        else:
            cookie_controller.set(
                "documind_supabase_session",
                result.session.refresh_token,
                max_age=30 * 24 * 60 * 60,
            )
            _apply_authenticated_supabase_user(result.user, result.session)
            st.rerun()

    with card:
        st.markdown('<div class="auth-divider">OR</div>', unsafe_allow_html=True)
        if st.button("Continue with Google", use_container_width=True, key="google_login"):
            if "auth" not in st.secrets:
                st.error(
                    "Google Sign-In is not configured. Add the [auth] section "
                    "to .streamlit/secrets.toml first."
                )
            else:
                st.login()
        st.markdown(
            '<div class="auth-footnote">Don\'t have an account?</div>',
            unsafe_allow_html=True,
        )
        if st.button("Sign Up", use_container_width=True, key="open_signup"):
            st.session_state.auth_view = "signup"
            st.rerun()
    return False


def handle_logout(cookie_controller):
    try:
        get_supabase_client().auth.sign_out()
    except Exception:
        pass
    cookie_controller.delete_cookie()
    save_current_user_state()
    clear_active_user_state()
    st.session_state.auth_method = None
    st.session_state.supabase_session = None
    for key in ("current_username", "current_user_id", "current_user"):
        st.session_state.pop(key, None)


def filter_retrieved_documents(retrieved_docs, selected_document, compare_documents=None):
    """Enforce the selected-PDF boundary before documents reach the QA chain."""
    if compare_documents:
        allowed_sources = set(compare_documents)
        return [
            doc for doc in retrieved_docs
            if (getattr(doc, "metadata", {}) or {}).get("source") in allowed_sources
        ]

    if selected_document == "All Documents":
        return retrieved_docs

    return [
        doc for doc in retrieved_docs
        if (getattr(doc, "metadata", {}) or {}).get("source") == selected_document
    ]


def truncate_evidence(evidence: str) -> str:
    """Limit only displayed evidence; the full chunk remains available to the LLM."""
    if len(evidence) <= EVIDENCE_MAX_CHARS:
        return evidence
    return evidence[:EVIDENCE_MAX_CHARS - 1].rstrip() + "…"


def build_source_details(retrieved_docs):
    """Create citation labels and de-duplicated exact chunk evidence."""
    sources_list = []
    source_evidence = []
    seen_sources = set()
    seen_evidence = set()

    for doc in retrieved_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source", "Unknown file")
        page = metadata.get("page")
        # PyPDFLoader's page index is 0-based; show 1-based to users.
        page_display = page + 1 if isinstance(page, int) else "Unknown"
        source_key = (source, page_display)

        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources_list.append(f"📄 {source} — Page {page_display}")

        evidence = getattr(doc, "page_content", "").strip()
        evidence_key = (source, page_display, evidence)
        if evidence and evidence_key not in seen_evidence:
            seen_evidence.add(evidence_key)
            source_evidence.append({
                "source": source,
                "page_display": page_display,
                "content": truncate_evidence(evidence),
            })

    return sources_list, source_evidence


def render_sources(sources_list, source_evidence=None):
    """Render normal citations plus the exact chunks the retriever returned."""
    for source in sources_list:
        st.write(source)

    if source_evidence:
        st.markdown("**Exact retrieved evidence**")
        for evidence in source_evidence:
            st.write(f"📄 {evidence['source']} — Page {evidence['page_display']}")
            st.markdown(
                "> " + evidence["content"].replace("\n", "\n> ")
            )


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


@st.cache_resource
def get_reranker():
    if CrossEncoder is None:
        return None
    try:
        return CrossEncoder(RERANKER_MODEL)
    except Exception:
        return None


def rewrite_query(llm, question, chat_history):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the latest question as a standalone search query. Use the "
            "conversation only when needed to resolve references such as it, they, "
            "or the risks. Expand the query with a few relevant synonyms, aliases, "
            "and section terminology that may appear in a document, while keeping "
            "the original meaning precise. Preserve names, numbers, quoted phrases, "
            "and the user's intent. Return only one concise search query."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "Latest question: {question}"),
    ])
    try:
        response = llm.invoke(prompt.format_messages(
            chat_history=chat_history,
            question=question,
        ))
        rewritten = getattr(response, "content", str(response)).strip()
        return rewritten or question
    except Exception:
        return question


def retrieve_hybrid(query, retrieval_state, mode, selected_document, document_a, document_b):
    source_indexes = retrieval_state.get("source_indexes", {})
    if mode == "Compare":
        allowed_sources = [document_a, document_b]
    elif selected_document == "All Documents":
        allowed_sources = list(source_indexes)
    else:
        allowed_sources = [selected_document]

    reranker = get_reranker()
    candidates = {}

    for source in allowed_sources:
        source_index = source_indexes.get(source)
        if source_index is None:
            continue

        def add_candidate(document, method, bm25_score=None):
            metadata = getattr(document, "metadata", {}) or {}
            key = (
                source,
                metadata.get("page"),
                getattr(document, "page_content", ""),
            )
            candidate = candidates.setdefault(
                key,
                {
                    "doc": document,
                    "methods": set(),
                    "bm25_score": None,
                    "rerank_score": None,
                },
            )
            candidate["methods"].add(method)
            if bm25_score is not None:
                candidate["bm25_score"] = float(bm25_score)

        for document in source_index["faiss"].invoke(query)[:FAISS_K]:
            add_candidate(document, "FAISS")

        if source_index["bm25"] is not None:
            scores = source_index["bm25"].get_scores(tokenize(query))
            ranked_indexes = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )[:BM25_K]
            for index in ranked_indexes:
                add_candidate(
                    source_index["documents"][index],
                    "BM25",
                    scores[index],
                )

    candidates_list = list(candidates.values())
    if reranker is not None and candidates_list:
        pairs = [(query, item["doc"].page_content) for item in candidates_list]
        try:
            scores = reranker.predict(pairs)
            for item, score in zip(candidates_list, scores):
                item["rerank_score"] = float(score)
            candidates_list.sort(
                key=lambda item: item["rerank_score"],
                reverse=True,
            )
        except Exception:
            reranker = None

    if reranker is None:
        candidates_list.sort(
            key=lambda item: (
                len(item["methods"]),
                item["bm25_score"] is not None,
                item["bm25_score"] or 0,
            ),
            reverse=True,
        )

    final_docs = []
    details = []
    for item in candidates_list[:RERANK_K]:
        document = item["doc"]
        final_docs.append(document)
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("source", "Unknown file")
        page = metadata.get("page")
        page_display = page + 1 if isinstance(page, int) else "Unknown"
        details.append({
            "source": source,
            "page_display": page_display,
            "chunk": metadata.get("chunk", "N/A"),
            "method": " + ".join(sorted(item["methods"])),
            "bm25_score": item["bm25_score"],
            "rerank_score": item["rerank_score"],
            "preview": document.page_content[:240].replace("\n", " "),
        })

    return final_docs, details


def render_retrieval_details(details):
    with st.expander("🔍 Retrieval Details", expanded=False):
        if not details:
            st.write("No retrieved chunks.")
            return
        for detail in details:
            bm25_score = (
                f"{detail['bm25_score']:.3f}"
                if detail["bm25_score"] is not None else "N/A"
            )
            rerank_score = (
                f"{detail['rerank_score']:.3f}"
                if detail["rerank_score"] is not None else "N/A"
            )
            st.markdown(
                f"**{detail['source']}** — Page {detail['page_display']} — "
                f"Chunk {detail['chunk']}  \n"
                f"Method: {detail['method']} | BM25: {bm25_score} | "
                f"Rerank: {rerank_score}  \n"
                f"{detail['preview']}"
            )


def verify_citations(llm, answer, retrieved_docs):
    """Verify answer claims against only the chunks retrieved for this answer."""
    if not retrieved_docs or not answer.strip():
        return {"status": "no_claims", "claims": []}

    evidence_blocks = []
    for index, document in enumerate(retrieved_docs, start=1):
        metadata = getattr(document, "metadata", {}) or {}
        page = metadata.get("page")
        page_display = page + 1 if isinstance(page, int) else "Unknown"
        evidence_blocks.append(
            f"[E{index}] Source: {metadata.get('source', 'Unknown file')} | "
            f"Page: {page_display}\n{document.page_content}"
        )

    verification_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You verify factual claims in an answer using ONLY the supplied PDF "
            "evidence. Ignore headings, transitions, and generic conversational "
            "phrases. Extract meaningful factual claims, especially numbers, dates, "
            "names, percentages, and comparisons. A claim is supported only when "
            "the evidence directly supports all important details. Use status values "
            "exactly: supported, partially_supported, unsupported. Never use outside "
            "knowledge. Return JSON only with this shape: {{\"claims\":[{{\"claim\": "
            "\"...\",\"status\":\"supported|partially_supported|unsupported\","
            "\"evidence_ids\":[\"E1\"],\"reason\":\"...\"}}]}}"
        ),
        (
            "human",
            "Answer:\n{answer}\n\nRetrieved PDF evidence:\n{evidence}"
        ),
    ])

    try:
        response = llm.invoke(verification_prompt.format_messages(
            answer=answer,
            evidence="\n\n".join(evidence_blocks),
        ))
        raw_result = getattr(response, "content", str(response)).strip()
        raw_result = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_result)
        result = json.loads(raw_result)
        raw_claims = result.get("claims", [])
        claims = []
        valid_ids = {f"E{index}" for index in range(1, len(retrieved_docs) + 1)}

        for raw_claim in raw_claims:
            claim = str(raw_claim.get("claim", "")).strip()
            if not claim:
                continue
            status = raw_claim.get("status")
            if status == "partially_supported":
                display_status = "⚠️ Partially supported"
            elif status == "supported":
                display_status = "✅ Supported"
            else:
                status = "unsupported"
                display_status = "❌ Unsupported"

            evidence_ids = [
                evidence_id for evidence_id in raw_claim.get("evidence_ids", [])
                if evidence_id in valid_ids
            ]
            if status in {"supported", "partially_supported"} and not evidence_ids:
                status = "unsupported"
                display_status = "❌ Unsupported"

            claim_evidence = []
            for evidence_id in evidence_ids:
                document = retrieved_docs[int(evidence_id[1:]) - 1]
                metadata = getattr(document, "metadata", {}) or {}
                page = metadata.get("page")
                claim_evidence.append({
                    "source": metadata.get("source", "Unknown file"),
                    "page": page + 1 if isinstance(page, int) else "Unknown",
                    "content": document.page_content,
                })
            claims.append({
                "claim": claim,
                "status": status,
                "display_status": display_status,
                "reason": str(raw_claim.get("reason", "")).strip(),
                "evidence": claim_evidence,
            })

        if not claims:
            return {"status": "no_claims", "claims": []}
        if any(claim["status"] == "unsupported" for claim in claims):
            overall = "🔴 Unsupported claims detected"
        elif any(claim["status"] == "partially_supported" for claim in claims):
            overall = "🟡 Partially supported"
        else:
            overall = "🟢 Fully supported"
        return {"status": "verified", "overall": overall, "claims": claims}
    except Exception:
        return {"status": "unavailable", "claims": []}


def render_citation_verification(verification):
    with st.expander("🔗 Citation Verification", expanded=False):
        if verification.get("status") == "unavailable":
            st.warning("⚠️ Citation verification unavailable for this response.")
            return
        if verification.get("status") == "no_claims":
            st.write("No factual claims to verify.")
            return
        st.markdown(f"**Overall:** {verification['overall']}")
        for index, claim in enumerate(verification["claims"], start=1):
            st.markdown(f"**Claim {index}:** {claim['claim']}")
            st.markdown(f"**Status:** {claim['display_status']}")
            if claim.get("reason"):
                st.caption(claim["reason"])
            for evidence in claim.get("evidence", []):
                st.markdown(
                    f"**Source:** {evidence['source']} — Page {evidence['page']}"
                )
                st.markdown(f"> {evidence['content'].replace(chr(10), chr(10) + '> ')}")


def retrieval_quality(
    query,
    details,
    mode,
    selected_document,
    document_a,
    document_b,
):
    """Classify retrieval support, not factual answer accuracy."""
    if not details:
        return "Weak / insufficient retrieval"

    stop_words = {
        "a", "about", "an", "and", "are", "as", "at", "by", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
        "this", "to", "was", "what", "when", "where", "which", "who", "why",
        "with",
    }
    query_terms = {
        term for term in tokenize(query)
        if len(term) > 2 and term not in stop_words
    }
    overlap_counts = []
    for detail in details:
        preview_terms = set(tokenize(detail.get("preview", "")))
        overlap_counts.append(len(query_terms & preview_terms))

    meaningful_overlap = max(overlap_counts, default=0)
    bm25_scores = [
        score for score in (
            detail.get("bm25_score") for detail in details
        )
        if score is not None
    ]
    rerank_scores = [
        score for score in (
            detail.get("rerank_score") for detail in details
        )
        if score is not None
    ]

    # CrossEncoder scores are the strongest available signal. The thresholds
    # are retrieval heuristics only; they are not factual confidence values.
    if rerank_scores:
        best_rerank = max(rerank_scores)
        if best_rerank >= 0.5 and meaningful_overlap > 0:
            return "Strong retrieval"
        if best_rerank >= 0 and meaningful_overlap > 0:
            return "Moderate retrieval"
        return "Weak / insufficient retrieval"

    # Without a reranker, require both keyword evidence and a positive BM25
    # signal so unrelated fallback chunks are not called relevant.
    best_bm25 = max(bm25_scores, default=0)
    source_count = len({detail["source"] for detail in details})
    expected_sources = (
        [document_a, document_b]
        if mode == "Compare"
        else ([selected_document] if selected_document != "All Documents" else None)
    )
    if meaningful_overlap == 0 or best_bm25 <= 0:
        return "Weak / insufficient retrieval"
    if expected_sources and source_count == len(expected_sources) and len(details) >= 2:
        return "Strong retrieval"
    if len(details) >= 2:
        return "Moderate retrieval"
    return "Moderate retrieval"


def build_export_content(question, answer, sources_list, source_evidence, markdown=False):
    heading = "# DocuMind Answer" if markdown else "DocuMind Answer"
    lines = [heading, "", "Question:", question, "", "Answer:", answer, "", "Sources:"]
    lines.extend(f"- {source}" for source in sources_list)
    lines.extend(["", "Evidence:"])
    for evidence in source_evidence:
        lines.extend([
            f"- {evidence['source']} — Page {evidence['page_display']}",
            evidence["content"],
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def render_export_buttons(question, answer, sources_list, source_evidence, key):
    txt_content = build_export_content(
        question, answer, sources_list, source_evidence
    )
    markdown_content = build_export_content(
        question, answer, sources_list, source_evidence, markdown=True
    )
    st.markdown("**⬇ Export Answer**")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download TXT",
            txt_content,
            file_name="pdf_answer.txt",
            mime="text/plain",
            key=f"download_txt_{key}",
        )
    with col2:
        st.download_button(
            "Download Markdown",
            markdown_content,
            file_name="pdf_answer.md",
            mime="text/markdown",
            key=f"download_md_{key}",
        )


cookie_controller = get_cookie_controller()
if not render_authentication(cookie_controller):
    st.stop()


# =====================================================
# SIDEBAR — always rendered first, before any model /
# API / PDF work, so the UI is never blank even if the
# heavy stuff below fails.
# =====================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">📚 DocuMind</div>'
        '<div class="sidebar-descriptor">PDF Intelligence Assistant</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Welcome back, {st.session_state.current_user} 👋")
    st.caption("Upload PDF documents and have an intelligent conversation with them.")
    st.divider()

    st.markdown("### 💬 Conversation")
    session_id = st.session_state.session_id
    st.selectbox("Mode", ["Chat", "Compare"], key="mode")
    conversation_key = get_conversation_key(
        session_id,
        st.session_state.mode,
        st.session_state.selected_document,
        st.session_state.compare_document_a,
        st.session_state.compare_document_b,
    )

    if conversation_key not in st.session_state.messages_by_session:
        st.session_state.messages_by_session[conversation_key] = []

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.store[conversation_key] = ChatMessageHistory()
        st.session_state.messages_by_session[conversation_key] = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    st.markdown("### 📂 Upload Documents")

    def _handle_file_selection():
        # Fires the instant Streamlit's server registers the committed
        # upload (mobile or desktop) and writes straight into
        # uploaded_files, instead of relying on the next full-body rerun
        # to notice a stale local `selected_files` variable — that gap is
        # the actual root cause of needing a second tap on mobile browsers
        # (see the chat explanation of the fix).
        files = st.session_state.get("pdf_uploader_widget") or []
        st.session_state.uploaded_files = files
        st.session_state.upload_just_received = bool(files)

    st.file_uploader(
        "Choose PDF file(s)",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader_widget",
        on_change=_handle_file_selection,
    )
    if st.session_state.get("upload_just_received"):
        st.info("📄 File(s) received — processing will start below.")
        st.session_state.upload_just_received = False
    uploaded_files = st.session_state.uploaded_files

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} PDF(s) selected")
        with st.expander("📄 View Selected Files", expanded=False):
            for uploaded_file in uploaded_files:
                st.caption(f"📄 {uploaded_file.name}")

        document_options = ["All Documents"] + [
            uploaded_file.name for uploaded_file in uploaded_files
        ]
        if st.session_state.selected_document not in document_options:
            st.session_state.selected_document = "All Documents"
        st.selectbox(
            "Ask about:",
            document_options,
            key="selected_document"
        )

        if len(document_options) >= 3:
            available_documents = document_options[1:]
            if st.session_state.compare_document_a not in available_documents:
                st.session_state.compare_document_a = available_documents[0]
            if (
                st.session_state.compare_document_b not in available_documents
                or st.session_state.compare_document_b == st.session_state.compare_document_a
            ):
                st.session_state.compare_document_b = next(
                    document for document in available_documents
                    if document != st.session_state.compare_document_a
                )
            if st.session_state.mode == "Compare":
                st.selectbox(
                    "Document A",
                    available_documents,
                    key="compare_document_a"
                )
                st.selectbox(
                    "Document B",
                    available_documents,
                    key="compare_document_b"
                )
        elif st.session_state.mode == "Compare":
            st.info("Upload at least two PDFs to use Compare mode.")
    else:
        st.info("Upload PDF files to begin.")

    st.divider()
    st.markdown("### ⚙️ Powered By")
    st.caption("🦜 LangChain")
    st.caption("🔍 FAISS")
    st.caption("🤗 HuggingFace")
    st.caption("⚡ Groq")
    st.caption("🎈 Streamlit")
    st.divider()

    st.markdown('<div class="made-by">Made with ❤️ by <b>Varsha Chauhan</b></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown(
        '<div class="sidebar-user-card">'
        '<div class="sidebar-user-label">Logged in as</div>'
        f'<div class="sidebar-user-name">{st.session_state.current_user}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    _questions_used_today = get_user_usage_count(st.session_state.get("current_username", ""))
    st.caption(
        f"💬 {_questions_used_today}/{USER_DAILY_QUESTION_LIMIT} questions used today"
    )
    if st.session_state.auth_method == "google":
        if st.button("Logout", key="google_logout", use_container_width=True):
            handle_logout(cookie_controller)
            st.logout()
    else:
        if st.button("Logout", key="pdf_logout", use_container_width=True):
            handle_logout(cookie_controller)
            st.rerun()

ensure_app_state_defaults()

conversation_key = get_conversation_key(
    session_id,
    st.session_state.mode,
    st.session_state.selected_document,
    st.session_state.compare_document_a,
    st.session_state.compare_document_b,
)
if conversation_key not in st.session_state.messages_by_session:
    st.session_state.messages_by_session[conversation_key] = []

# =====================================================
# MAIN HEADER — always rendered, regardless of whether
# imports, the API key, or the model are available.
# =====================================================
st.markdown(
    '<div class="hero-container">'
    '<div class="hero-title">Chat with your documents.</div>'
    '<div class="hero-subtitle">RAG-powered PDF intelligence for grounded, '
    'context-aware conversations.</div>'
    '<div class="hero-description">Upload PDFs, ask questions, compare documents, '
    'and get answers grounded in your files using hybrid retrieval, reranking, '
    'and citation verification.</div>'
    '<div class="hero-capabilities">RAG &nbsp;•&nbsp; Hybrid Retrieval &nbsp;•&nbsp; '
    'Reranking &nbsp;•&nbsp; Citation Verification</div>'
    '</div>',
    unsafe_allow_html=True
)

# =====================================================
# From here on, everything that can fail (imports, the
# API key, model init, PDF processing) is checked AFTER
# the sidebar and header already exist on screen, and
# every failure produces a clear, specific message
# instead of a blank page or a swallowed exception.
# =====================================================
if not IMPORTS_OK:
    st.error("❌ Failed to import required packages.")
    st.exception(IMPORT_ERROR)
    st.info(
        "Run this in your terminal (in the same environment you use to "
        "launch Streamlit) and check the versions:\n\n"
        "```\npip show langchain-core langchain-community langchain-classic "
        "langchain-groq langchain-huggingface langchain-text-splitters "
        "faiss-cpu pypdf\n```\n\n"
        "If a package is missing, install it with `pip install <name>`."
    )
    st.stop()

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("❌ GROQ_API_KEY was not found in `.streamlit/secrets.toml`.")
    st.info(
        "Create a file at `.streamlit/secrets.toml` next to `app.py` with:\n\n"
        '```\nGROQ_API_KEY = "your-key-here"\n```'
    )
    st.stop()


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource
def get_llm(_api_key):
    return ChatGroq(api_key=_api_key, model=MODEL_NAME, streaming=True)


# =====================================================
# MAIN BODY
# =====================================================
documents_are_ready = (
    st.session_state.history_aware_retriever is not None
    and st.session_state.question_answer_chain is not None
)

if uploaded_files or documents_are_ready:

    current_files_fingerprint = (
        tuple(sorted((f.name, hash_file(f)) for f in uploaded_files))
        if uploaded_files
        else st.session_state.processed_files_fingerprint
    )

    if st.session_state.processed_files_fingerprint != current_files_fingerprint:
        files_were_already_processed = st.session_state.processed_files_fingerprint is not None

        with st.spinner("🤖 Connecting to the Groq model..."):
            try:
                llm = get_llm(api_key)
            except Exception as e:
                st.error("❌ Failed to initialize the Groq LLM. Check your API key and model name.")
                st.exception(e)
                st.stop()

        with st.spinner("📚 Reading your PDF documents..."):
            documents = []
            total_pages = 0
            unreadable_files = []

            for uploaded_file in uploaded_files:
                temp_pdf_path = f"temp_{uuid.uuid4().hex}.pdf"
                try:
                    with open(temp_pdf_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getvalue())

                    loader = PyPDFLoader(temp_pdf_path)
                    docs = loader.load()

                    extracted_chars = sum(len(d.page_content.strip()) for d in docs)
                    if extracted_chars == 0:
                        st.warning(
                            f"⚠️ '{uploaded_file.name}' produced no extractable text. "
                            "It may be a scanned/image-only PDF, which this app can't OCR."
                        )

                    total_pages += len(docs)
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    documents.extend(docs)

                except Exception as e:
                    unreadable_files.append(uploaded_file.name)
                    st.warning(f"⚠️ Could not read '{uploaded_file.name}': {e}")
                finally:
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)

            if not documents:
                st.error("❌ No readable content was found in the uploaded PDF(s).")
                st.stop()

        with st.spinner("🧩 Splitting text into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)

            if not splits:
                st.error("❌ Documents were loaded but produced no text chunks to index.")
                st.stop()

        with st.spinner("🔎 Generating embeddings and building the search index..."):
            try:
                embeddings = get_embeddings()
                splits_by_source = {}
                for split in splits:
                    source = split.metadata.get("source", "Unknown file")
                    page = split.metadata.get("page")
                    split.metadata["page_display"] = (
                        page + 1 if isinstance(page, int) else "Unknown"
                    )
                    splits_by_source.setdefault(source, []).append(split)

                source_indexes = {}
                document_analytics = {}
                for source, source_splits in splits_by_source.items():
                    source_store = FAISS.from_documents(source_splits, embedding=embeddings)
                    faiss_retriever = source_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": min(RETRIEVER_K, len(source_splits)),
                            "fetch_k": min(RETRIEVER_FETCH_K, len(source_splits)),
                            "lambda_mult": 0.5
                        }
                    )
                    bm25 = None
                    if BM25Okapi is not None:
                        bm25 = BM25Okapi([
                            tokenize(split.page_content)
                            for split in source_splits
                        ])
                    source_indexes[source] = {
                        "faiss": faiss_retriever,
                        "bm25": bm25,
                        "documents": source_splits,
                    }
                    document_analytics[source] = {
                        "pages": len({
                            document.metadata.get("page")
                            for document in documents
                            if document.metadata.get("source") == source
                        }),
                        "chunks": len(source_splits),
                    }

                def retrieve_from_each_source(query):
                    return retrieve_hybrid(
                        query,
                        {"source_indexes": source_indexes},
                        st.session_state.get("mode", "Chat"),
                        st.session_state.get("selected_document", "All Documents"),
                        st.session_state.get("compare_document_a", ""),
                        st.session_state.get("compare_document_b", ""),
                    )[0]

                retriever = RunnableLambda(retrieve_from_each_source)
            except Exception as e:
                st.error("❌ Failed to build the embedding index (HuggingFace/FAISS).")
                st.exception(e)
                st.stop()

            contextualize_system_prompt = (
                "Given the chat history and the latest user question, determine "
                "whether the latest question refers to something mentioned earlier "
                "in the conversation. If necessary, reformulate it into a standalone "
                "question. Do not answer the question. Return only the standalone question."
            )
            contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_prompt
            )

            system_prompt = (
                "You are a helpful PDF assistant. Answer the user's question using "
                "ONLY the information available in the provided PDF context. "
                "Do not invent information. If the answer is not present in the "
                "context, clearly say that you could not find it in the uploaded "
                "PDFs. Keep your answers clear, accurate, well-structured, and easy "
                "to understand. Use Markdown for formatting and never output HTML "
                "tags such as <br>. Each context block includes its source file and "
                "page. Keep facts tied to that source: never attribute a fact from "
                "one company or PDF to another. For comparison questions, answer "
                "each company separately. If a company has no explicit evidence "
                "for a requested category in its source context, say so instead of "
                "borrowing facts from another company. Format the answer with a "
                "clear heading for each company and short bullet points. Do not "
                "use Markdown tables, numbered lists that continue across companies, "
                "or raw source-file labels in the answer; source details are shown "
                "separately below the response. Do not use outside knowledge or "
                "guess when the context does not support an answer. If the answer "
                "cannot be supported, say: I couldn't find enough information in "
                "the uploaded PDFs to answer that.\n\nContext:\n{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            document_prompt = ChatPromptTemplate.from_messages([
                ("human", "Source file: {source}\nPage: {page_display}\nContent:\n{page_content}")
            ])
            question_answer_chain = create_stuff_documents_chain(
                llm,
                qa_prompt,
                document_prompt=document_prompt
            )

            if files_were_already_processed:
                st.session_state.store[conversation_key] = ChatMessageHistory()
                st.session_state.messages_by_session[conversation_key] = []

            st.session_state.processed_files_fingerprint = current_files_fingerprint
            st.session_state.history_aware_retriever = history_aware_retriever
            st.session_state.question_answer_chain = question_answer_chain
            st.session_state.retrieval_state = {
                "source_indexes": source_indexes,
            }
            st.session_state.document_analytics = document_analytics
            st.session_state.pdf_info = {
                "files": len(uploaded_files),
                "pages": total_pages,
                "chunks": len(splits)
            }

        st.success("🎉 Your documents are ready! Start asking questions.")

    history_aware_retriever = st.session_state.history_aware_retriever
    question_answer_chain = st.session_state.question_answer_chain

    if history_aware_retriever is None or question_answer_chain is None:
        st.info("Upload PDF(s) above to build the search index before chatting.")
        st.stop()

    pdf_info = st.session_state.pdf_info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 PDFs", pdf_info["files"])
    with col2:
        st.metric("📑 Pages", pdf_info["pages"])
    with col3:
        st.metric("🧩 Chunks", pdf_info["chunks"])

    with st.expander("📊 Document Analytics"):
        analytics = st.session_state.document_analytics
        st.write(f"**Total Documents:** {len(analytics)}")
        st.write(f"**Total Pages:** {sum(item['pages'] for item in analytics.values())}")
        st.write(f"**Total Chunks:** {sum(item['chunks'] for item in analytics.values())}")
        for source, item in analytics.items():
            st.markdown(
                f"**{source}**\n- Pages: {item['pages']}\n- Chunks: {item['chunks']}"
            )

    st.divider()
    st.markdown("## 💬 Chat with your Documents")
    st.caption("Ask questions based on the content of your uploaded PDFs.")

    messages = st.session_state.messages_by_session[conversation_key]

    if not messages:
        st.markdown(
            '<div class="welcome-card">'
            '<div class="welcome-icon">🎉</div>'
            '<div class="welcome-title">Your PDFs are ready!</div>'
            '<div class="welcome-text">You can now ask questions, request summaries, '
            'explore concepts, and understand your documents through intelligent '
            'conversations.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.divider()

    if st.session_state.mode == "Compare":
        compare_question = st.text_input(
            "Question",
            key="compare_question",
            placeholder="Compare their financial performance."
        )
        if st.button("⚖️ Compare Documents", use_container_width=True):
            if compare_question.strip():
                st.session_state.pending_question = compare_question.strip()
                st.rerun()

    st.markdown("### 💡 Try asking")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 Summarize the PDFs", use_container_width=True):
            st.session_state.pending_question = "Summarize the uploaded documents."
            st.rerun()
    with col2:
        if st.button("🔑 Main Concepts", use_container_width=True):
            st.session_state.pending_question = (
                "What are the main concepts discussed in the uploaded PDFs?"
            )
            st.rerun()
    with col3:
        if st.button("📚 Explain Simply", use_container_width=True):
            st.session_state.pending_question = "Explain the main topic in simple words."
            st.rerun()

    for message_index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(normalize_answer(message["content"]))
            if message.get("sources"):
                with st.expander("📄 View Sources"):
                    render_sources(
                        message["sources"],
                        message.get("source_evidence")
                    )
            if message["role"] == "assistant":
                previous_message = messages[message_index - 1] if message_index else {}
                render_export_buttons(
                    previous_message.get("content", ""),
                    message["content"],
                    message.get("sources", []),
                    message.get("source_evidence", []),
                    f"history_{message_index}",
                )
                if message.get("retrieval_details"):
                    render_retrieval_details(message["retrieval_details"])
                    st.caption(
                        "Retrieval quality signal: "
                        + message.get("retrieval_quality", "N/A")
                        + " (not factual accuracy)"
                    )
                if message.get("citation_verification"):
                    render_citation_verification(message["citation_verification"])

    typed_question = (
        st.chat_input("Ask anything about your PDFs...")
        if st.session_state.mode == "Chat"
        else None
    )

    user_question = None
    if st.session_state.pending_question is not None:
        user_question = st.session_state.pending_question
        st.session_state.pending_question = None
    elif typed_question:
        user_question = typed_question

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        messages.append({"role": "user", "content": user_question})

        user_key = st.session_state.get("current_username", "")

        if not user_has_remaining_quota(user_key):
            limit_message = (
                f"⏳ You've reached your daily limit of {USER_DAILY_QUESTION_LIMIT} "
                "questions. Please try again tomorrow."
            )
            with st.chat_message("assistant"):
                st.warning(limit_message)
            messages.append({"role": "assistant", "content": limit_message})
            st.stop()

        # Reserve this question against the user's daily allowance up front.
        # If the request turns out to fail because of a shared provider-side
        # outage (not this user's fault), it's refunded below.
        increment_user_usage(user_key)

        session_history = get_session_history(conversation_key)

        try:
            with st.spinner("🔎 Searching your documents..."):
                llm = get_llm(api_key)
                retrieval_query = rewrite_query(
                    llm,
                    user_question,
                    session_history.messages,
                )
                retrieved_docs, retrieval_details = retrieve_hybrid(
                    retrieval_query,
                    st.session_state.retrieval_state,
                    st.session_state.mode,
                    st.session_state.get("selected_document", "All Documents"),
                    st.session_state.get("compare_document_a", ""),
                    st.session_state.get("compare_document_b", ""),
                )
                retrieved_docs = filter_retrieved_documents(
                    retrieved_docs,
                    st.session_state.get("selected_document", "All Documents"),
                    (
                        st.session_state.get("compare_document_a"),
                        st.session_state.get("compare_document_b"),
                    )
                    if st.session_state.get("mode") == "Compare"
                    else None,
                )
                retrieval_details = [
                    detail for detail in retrieval_details
                    if detail["source"] in {
                        getattr(doc, "metadata", {}).get("source")
                        for doc in retrieved_docs
                    }
                ]
        except Exception as e:
            with st.chat_message("assistant"):
                if is_provider_quota_error(e):
                    refund_user_usage(user_key)
                    st.error(
                        "⚠️ AI service is temporarily unavailable due to provider "
                        "limits. Please try again later."
                    )
                else:
                    st.error("❌ The retriever failed while searching your documents.")
                    st.exception(e)
            st.stop()

        with st.chat_message("assistant"):
            if not retrieved_docs:
                answer = "I couldn't find relevant information about this question in the uploaded PDFs."
                sources_list = []
                source_evidence = []
                st.markdown(answer)
            else:
                try:
                    with st.spinner("🤖 Thinking..."):
                        answer = ""
                        answer_placeholder = st.empty()
                        for chunk in question_answer_chain.stream({
                            "input": user_question,
                            "chat_history": session_history.messages,
                            "context": retrieved_docs
                        }):
                            answer += chunk
                            answer_placeholder.markdown(
                                f"{normalize_answer(answer)}▌"
                            )
                        answer = normalize_answer(answer)
                        answer_placeholder.markdown(answer)
                except Exception as e:
                    if is_provider_quota_error(e):
                        refund_user_usage(user_key)
                        st.error(
                            "⚠️ AI service is temporarily unavailable due to provider "
                            "limits. Please try again later."
                        )
                    else:
                        st.error("❌ The Groq model failed to generate an answer.")
                        st.exception(e)
                    st.stop()

                sources_list, source_evidence = build_source_details(retrieved_docs)

                if sources_list:
                    with st.expander("📄 View Sources"):
                        render_sources(sources_list, source_evidence)

            citation_verification = verify_citations(
                llm,
                answer,
                retrieved_docs,
            )

            render_export_buttons(
                user_question,
                answer,
                sources_list,
                source_evidence,
                f"current_{len(messages)}",
            )
            render_retrieval_details(retrieval_details)
            retrieval_quality_label = retrieval_quality(
                retrieval_query,
                retrieval_details,
                st.session_state.mode,
                st.session_state.get("selected_document", "All Documents"),
                st.session_state.get("compare_document_a", ""),
                st.session_state.get("compare_document_b", ""),
            )
            st.caption(
                f"Retrieval quality signal: {retrieval_quality_label} "
                "(not factual accuracy)"
            )
            render_citation_verification(citation_verification)

        session_history.add_user_message(user_question)
        session_history.add_ai_message(answer)
        messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_list,
            "source_evidence": source_evidence,
            "retrieval_details": retrieval_details,
            "retrieval_quality": retrieval_quality_label,
            "citation_verification": citation_verification,
        })

else:
    st.markdown(
        '<div class="welcome-card">'
        '<div class="welcome-icon">📚</div>'
        '<div class="welcome-title">Start by uploading your PDFs</div>'
        '<div class="welcome-text">Upload one or multiple PDF documents from the '
        'sidebar and start asking intelligent, context-based questions.</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("### ✨ How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="info-card"><div class="info-icon">'
            '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M5 15v4h14v-4"/></svg>'
            '</div>'
            '<div class="info-title">1. Upload</div>'
            '<div class="info-text">Upload one or more PDF documents.</div></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div class="info-card"><div class="info-icon">'
            '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v3"/><path d="M12 18v3"/><path d="m4.2 4.2 2.1 2.1"/><path d="m17.7 17.7 2.1 2.1"/><path d="M3 12h3"/><path d="M18 12h3"/><path d="m4.2 19.8 2.1-2.1"/><path d="m17.7 6.3 2.1-2.1"/><circle cx="12" cy="12" r="4"/></svg>'
            '</div>'
            '<div class="info-title">2. Process</div>'
            '<div class="info-text">Documents are chunked and indexed for intelligent retrieval.</div></div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<div class="info-card"><div class="info-icon">'
            '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16v11H8l-4 4z"/><path d="M8 9h8"/><path d="M8 12h5"/></svg>'
            '</div>'
            '<div class="info-title">3. Chat</div>'
            '<div class="info-text">Ask questions and receive grounded answers with source references.</div></div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        '<div class="made-by">Source-grounded answers &nbsp; • &nbsp; '
        'Document comparison &nbsp; • &nbsp; Conversation memory &nbsp; • &nbsp; '
        'Multi-PDF analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="made-by">Made with ❤️ by <b>Varsha Chauhan</b></div>', unsafe_allow_html=True)
