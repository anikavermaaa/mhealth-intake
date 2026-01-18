# app.py
# -------------------------------------------
# Prereqs (PowerShell):
#   .\.venv\Scripts\Activate.ps1
#   pip install streamlit openai python-dotenv scikit-learn joblib torch transformers
# Add your key in configs/.env:  OPENAI_API_KEY=sk-...
# Run:
#   streamlit run .\app.py
# -------------------------------------------

import os, json, re
import streamlit as st
from dotenv import load_dotenv

# ============ Config & Keys ============
load_dotenv("configs/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============ Optional BERT dep/anx screening ============
try:
    from model_infer import TwoHeadInfer
    HAS_BERT_SCREEN = True
except Exception:
    HAS_BERT_SCREEN = False

@st.cache_resource
def load_screening_model():
    """Load trained two-head model from outputs/... (cached)."""
    # Adjust this path to your actual best checkpoint
    candidates = [
        "outputs/twohead/best.pt",
        "outputs/debug_run/best.pt",
        "outputs/best.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return TwoHeadInfer(p)
    raise FileNotFoundError(
        "Two-head checkpoint not found. Put best.pt in outputs/twohead/ or outputs/debug_run/."
    )

# ============ Root-cause ML (TF-IDF + OVR Logistic) ============
RC_AVAILABLE = True
try:
    from rc_infer import predict_root_causes
except Exception:
    RC_AVAILABLE = False

# ============ Streamlit Page ============
st.set_page_config(page_title="Procrastination Coach — Intake", page_icon="💬")
st.title("Procrastination Coach — Intake")

# ============ Session State ============
if "messages" not in st.session_state:
    st.session_state.messages = []              # list[(role,str)]
if "used_fups" not in st.session_state:
    st.session_state.used_fups = set()          # follow-ups asked already
if "pending_field" not in st.session_state:
    st.session_state.pending_field = None       # which slot the next user reply should fill
if "awaiting_confirm" not in st.session_state:
    st.session_state.awaiting_confirm = False   # waiting for yes/no on summary
if "finished" not in st.session_state:
    st.session_state.finished = False           # stop asking after confirmation
if "intake_json" not in st.session_state:
    st.session_state.intake_json = None         # final structured output
if "form" not in st.session_state:
    st.session_state.form = {
        "task_ctx": {"title": None, "deadline_text": None, "est_minutes": None},
        "feelings_words": [],
        "non_disorder_causes_selected": [],
        "sleep_hours": None,
        "energy_0_3": None,  # 0=very low,1=low,2=med,3=high
        "distractions": [],
        "coping_attempts": [],
        "anxiety_phrases": [],
        "depression_phrases": [],
        "adhd_phrases": [],
        "ocd_phrases": [],
        "values_or_goals": [],
        "crisis": False,
        "summary_one_line": ""
    }
form = st.session_state.form

# ============ Therapist Prompt ============
DEFAULT_THERAPIST_PROMPT = (
    "You are a warm, CBT-informed intake coach.\n"
    "Ask ONE short, natural question at a time (<=2 sentences). Prefer following up on the user's last message.\n"
    "Use brief reflections. No diagnosis or medical advice. Avoid PII.\n"
    "If self-harm/violence appears, say you'll show help resources and ask if they feel safe to continue.\n"
    "Stop after you present a short summary and the user confirms it."
)
try:
    INTAKE_SYSTEM_PROMPT = open("prompts/intake_system.txt", "r", encoding="utf-8").read().strip() or DEFAULT_THERAPIST_PROMPT
except FileNotFoundError:
    INTAKE_SYSTEM_PROMPT = DEFAULT_THERAPIST_PROMPT

# ============ Safety & Follow-ups ============
SAFETY = ["suicide", "kill myself", "self harm", "harm myself", "harm others"]
FOLLOWUPS = [
    {"id":"self_esteem",
     "triggers":["low self esteem","low self-esteem","feel worthless","not good enough","ashamed of myself"],
     "ask":["Thanks for sharing that. What kinds of thoughts make your self-esteem feel low?"]},
    {"id":"anxiety",
     "triggers":["racing thoughts","what if","spiral","panic","tight chest","cant switch off","can’t switch off"],
     "ask":["Do the worries show up more at night or during work time?"]},
    {"id":"sleep",
     "triggers":["didn't sleep","didnt sleep","couldn't sleep","couldnt sleep","sleep 3","sleep 4","insomnia","doomscroll"],
     "ask":["Roughly how many hours did you sleep last night?"]},
    {"id":"distraction",
     "triggers":["phone","notifications","noise","reels","social media","tabs"],
     "ask":["Which distraction hits you first, and could you remove just that one for 20 minutes?"]},
]
CORE_QUESTIONS = [
    ("task_ctx.title", "What one task is most affected by this?"),
    ("task_ctx.deadline_text", "When is that task due?"),
    ("task_ctx.est_minutes", "About how many focused minutes would it take?"),
    ("feelings_words", "How do you feel right now (sad, anxious, overwhelmed, tired, okay)? Pick any that fit."),
    ("sleep_hours", "How many hours did you sleep last night?"),
    ("energy_0_3", "Energy right now: low, medium, or high?"),
    ("non_disorder_causes_selected", "Which might fit: perfectionism, fear of failure, low interest, overwhelm/burnout, distractions?")
]

# ============ LLM Helper ============
def llm_chat(system_prompt: str, messages: list[dict], model: str = "gpt-4o-mini", max_tokens: int = 120) -> str:
    """Return one short assistant message. Raises if key/model missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in configs/.env")
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai SDK not installed. pip install openai") from e

    client = OpenAI(api_key=OPENAI_API_KEY)
    chat_messages = [{"role": "system", "content": system_prompt}]
    chat_messages += messages[-10:]  # keep recent turns

    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=max_tokens,
        messages=chat_messages,
    )
    return resp.choices[0].message.content.strip()

# ============ Helpers ============
def safety_scan(txt: str) -> bool:
    low = txt.lower()
    return any(k in low for k in SAFETY)

def find_followup(txt: str):
    low = txt.lower()
    for rule in FOLLOWUPS:
        if rule["id"] in st.session_state.used_fups:
            continue
        if any(trig in low for trig in rule["triggers"]):
            st.session_state.used_fups.add(rule["id"])
            return rule["ask"][0]
    return None

def get_field(path):
    cur = form
    for k in path.split("."):
        cur = cur.get(k, None) if isinstance(cur, dict) else None
    return cur

def set_field(path, value):
    parts = path.split(".")
    cur = form
    for k in parts[:-1]:
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = value

def ask(question, field=None):
    st.chat_message("assistant").markdown(question)
    st.session_state.messages.append(("assistant", question))
    st.session_state.pending_field = field

def next_missing_question():
    for path, question in CORE_QUESTIONS:
        val = get_field(path)
        if val in (None, "", []) or (isinstance(val, int) and val == 0):
            return path, question
    return None, None

def light_extract_free_text(text: str):
    low = text.lower()
    for w in ["sad","anxious","overwhelmed","tired","okay"]:
        if w in low and w not in form["feelings_words"]:
            form["feelings_words"].append(w)
    if "racing thoughts" in low or "what if" in low or "spiral" in low:
        if "racing thoughts/what-if" not in form["anxiety_phrases"]:
            form["anxiety_phrases"].append("racing thoughts/what-if")
    if "lost interest" in low or "nothing feels worth" in low:
        if "lost interest" not in form["depression_phrases"]:
            form["depression_phrases"].append("lost interest")
    if "forget" in low or "hard to start" in low or "lose track" in low:
        if "start/forget" not in form["adhd_phrases"]:
            form["adhd_phrases"].append("start/forget")
    if "intrusive thoughts" in low or "rechecking" in low:
        if "intrusive/rechecking" not in form["ocd_phrases"]:
            form["ocd_phrases"].append("intrusive/rechecking")

def assign_answer_to_field(field_path: str, user_text: str):
    low = user_text.strip().lower()

    if field_path == "task_ctx.title":
        set_field(field_path, user_text.strip()[:80])

    elif field_path == "task_ctx.deadline_text":
        set_field(field_path, user_text.strip()[:80])

    elif field_path == "task_ctx.est_minutes":
        m = re.search(r"(\d+)", low)
        if m: set_field(field_path, int(m.group(1)))

    elif field_path == "sleep_hours":
        h = re.search(r"(\d+)", low)
        if h: set_field(field_path, int(h.group(1)))

    elif field_path == "energy_0_3":
        if any(w in low for w in ["very low","exhausted"]):
            set_field(field_path, 0)
        elif "low" in low:
            set_field(field_path, 1)
        elif "medium" in low or "mid" in low or "okay" in low:
            set_field(field_path, 2)
        elif "high" in low:
            set_field(field_path, 3)

    elif field_path == "feelings_words":
        picked = []
        for w in ["sad","anxious","overwhelmed","tired","okay"]:
            if w in low: picked.append(w)
        if picked:
            set_field(field_path, list(sorted(set(form["feelings_words"] + picked))))

    elif field_path == "non_disorder_causes_selected":
        causes = []
        if "perfectionism" in low: causes.append("perfectionism")
        if "fear of failure" in low or "afraid to fail" in low: causes.append("fear of failure")
        if "low interest" in low or "bored" in low: causes.append("low interest")
        if "overwhelm" in low or "burnout" in low: causes.append("overwhelm/burnout")
        if "distraction" in low or "phone" in low or "notifications" in low: causes.append("distraction")
        if causes:
            set_field(field_path, list(sorted(set(form["non_disorder_causes_selected"] + causes))))

def all_basics_collected():
    needed = [
        form["task_ctx"]["title"],
        form["task_ctx"]["deadline_text"],
        form["task_ctx"]["est_minutes"],
        form["feelings_words"],
        form["sleep_hours"],
        form["energy_0_3"]
    ]
    return all(v not in (None, "", []) for v in needed)

def state_summary_for_llm() -> str:
    f = form
    return (
        "Captured so far:\n"
        f"- task: {f['task_ctx']['title']}\n"
        f"- deadline: {f['task_ctx']['deadline_text']}\n"
        f"- est_minutes: {f['task_ctx']['est_minutes']}\n"
        f"- feelings: {', '.join(f['feelings_words'])}\n"
        f"- sleep_hours: {f['sleep_hours']}\n"
        f"- energy_0_3: {f['energy_0_3']}\n"
        f"- causes: {', '.join(f['non_disorder_causes_selected'])}\n"
        "Rules: Ask ONE short question. Prefer following up on the last user message. "
        "If a required slot is missing, ask the best next question to fill it. No PII. No diagnosis.\n"
    )

def early_chat_text(max_user_msgs: int = 3) -> str:
    """Join the first few user messages to screen non-disorder causes."""
    user_msgs = [m for r, m in st.session_state.messages if r == "user"]
    return "\n".join(user_msgs[:max_user_msgs]).strip()

def convo_window_text(max_turns: int = 8, max_chars: int = 1800) -> str:
    """Join recent user+assistant turns with role tags; tail-truncate to ~BERT limit."""
    parts = []
    for role, msg in st.session_state.messages[-max_turns:]:
        tag = "USR" if role == "user" else "SYS"
        parts.append(f"[{tag}] {msg.strip()}")
    txt = "\n".join(parts).strip()
    # keep tail so the model sees most recent context
    return txt[-max_chars:]

# ============ Render history ============
for role, msg in st.session_state.messages:
    st.chat_message(role).markdown(msg)

# First greeting
if not st.session_state.messages:
    ask("Hi—what’s the issue today, and what feels hardest about it?", field=None)

# Stop if finished
if st.session_state.finished:
    st.info("Intake saved. Type **restart** to begin a new session.")
    st.stop()

# ============ Sidebar ============
with st.sidebar:
    st.subheader("Your info so far")
    st.write(f"**Task:** {form['task_ctx']['title'] or '—'}")
    st.write(f"**Deadline:** {form['task_ctx']['deadline_text'] or '—'}")
    st.write(f"**Minutes:** {form['task_ctx']['est_minutes'] or '—'}")
    st.write(f"**Feelings:** {', '.join(form['feelings_words']) or '—'}")
    st.write(f"**Sleep (h):** {form['sleep_hours'] or '—'}")
    st.write(f"**Energy (0-3):** {form['energy_0_3'] if form['energy_0_3'] is not None else '—'}")
    st.write(f"**Causes:** {', '.join(form['non_disorder_causes_selected']) or '—'}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Restart"):
            for k in list(st.session_state.keys()):
                if k not in ("_session_state",):
                    del st.session_state[k]
            st.rerun()
    with col2:
        if st.button("Show JSON"):
            st.session_state.intake_json = form

    if st.session_state.intake_json:
        with st.expander("View summary JSON"):
            st.json(st.session_state.intake_json)
        st.download_button(
            "Download intake.json",
            data=json.dumps(st.session_state.intake_json, indent=2),
            file_name="intake.json",
            mime="application/json"
        )

    # ------- BERT dep/anx screening (force last user msg only) -------
    st.markdown("---")
    st.subheader("Screening (disorders)")
    if not HAS_BERT_SCREEN:
        st.caption("Add model_infer.py and a trained best.pt in outputs/... to enable this.")
    else:
        run_screen = st.toggle("Run screening (last user message only)", value=False)

        if run_screen and st.session_state.messages:
            try:
                infer = load_screening_model()

                # Force: score only the last USER message
                last_user_texts = [m for r, m in st.session_state.messages if r == "user"]
                text_to_score = last_user_texts[-1].strip() if last_user_texts else ""

                if text_to_score:
                    probs = infer.predict(text_to_score)  # {'dep_prob': x, 'anx_prob': y}
                    st.write(f"**Depression-like signals:** {probs['dep_prob']:.2f}")
                    st.write(f"**Anxiety-like signals:** {probs['anx_prob']:.2f}")
                    with st.expander("See the exact text sent to the model"):
                        st.code(text_to_score)
                    st.caption("Not a diagnosis. Model output may be inaccurate.")
                else:
                    st.info("No user message to score yet.")
            except Exception as e:
                st.error(f"Screening error: {e}")

    # ------- Root causes (non-disorder) -------
    st.markdown("---")
    st.subheader("Root causes (non-disorder)")
    if not RC_AVAILABLE:
        st.caption("Add rc_outputs/ + rc_infer.py to enable non-disorder causes.")
    else:
        run_causes = st.toggle(
            "Analyze early chat",
            value=False,
            help="Uses the first few user messages to estimate non-disorder drivers."
        )
        if run_causes:
            blob = early_chat_text(max_user_msgs=3)
            if not blob:
                st.info("Start chatting first, then toggle this.")
            else:
                try:
                    rc = predict_root_causes(blob)  # {'proba': {...}, 'hits': [...]}
                    hits = rc["hits"]
                    st.write("**Likely drivers:** " + (", ".join(hits) if hits else "none at threshold"))
                    with st.expander("See probabilities"):
                        for k, v in rc["proba"].items():
                            st.write(f"- {k}: {v:.2f}")
                    st.caption("Multi-label; thresholds tuned per class on validation. Not a diagnosis.")
                except Exception as e:
                    st.error(f"Root-cause inference error: {e}")

st.caption("Tip: Use the sidebar toggles to screen the full conversation and estimate non-disorder drivers.")

# ============ Input handling ============
user = st.chat_input("Type here…")
if user:
    # quick text command to restart
    if user.strip().lower() == "restart":
        for k in list(st.session_state.keys()):
            if k not in ("_session_state",):
                del st.session_state[k]
        st.rerun()

    st.chat_message("user").markdown(user)
    st.session_state.messages.append(("user", user))

    # 1) safety first
    if safety_scan(user):
        form["crisis"] = True
        ask("Thanks for telling me. I can share crisis resources. Do you feel safe to continue here?", field=None)

    # 2) confirmation flow
    elif st.session_state.awaiting_confirm:
        ans = user.strip().lower()
        if ans in {"yes","y","yeah","yup","correct","ok","okay"}:
            st.success("Thanks — saved your summary.")
            st.session_state.finished = True
            st.session_state.awaiting_confirm = False
            st.session_state.intake_json = form
        elif ans in {"no","n","nope","not really"}:
            ask("No problem — tell me what I should correct (e.g., 'deadline is Friday', 'minutes 45').", field=None)
            st.session_state.awaiting_confirm = False
        else:
            ask("Please reply **yes** or **no**.", field=None)

    else:
        # 3) if a core question was pending, write this answer into that slot
        if st.session_state.pending_field:
            assign_answer_to_field(st.session_state.pending_field, user)
            st.session_state.pending_field = None

        # 4) light extraction from any free text
        light_extract_free_text(user)

        # 5) priority: follow-up on user’s last message (at most one)
        fup = find_followup(user)
        if fup:
            ask(fup, field=None)

        else:
            # 6) summarize when basics done, else ask next Q (LLM or fallback)
            if all_basics_collected():
                summary = (
                    f"Summary: task='{form['task_ctx']['title']}', "
                    f"due='{form['task_ctx']['deadline_text']}', "
                    f"minutes={form['task_ctx']['est_minutes']}, "
                    f"feelings={', '.join(form['feelings_words'])}, "
                    f"sleep={form['sleep_hours']}h, "
                    f"energy={form['energy_0_3']}, "
                    f"causes={', '.join(form['non_disorder_causes_selected']) or '—'}."
                )
                form["summary_one_line"] = summary[:120]
                st.session_state.awaiting_confirm = True
                ask(summary + " Did I get that right? If yes, type 'yes'.", field=None)
            else:
                try:
                    msg_history = [
                        {"role": ("assistant" if r == "assistant" else "user"), "content": m}
                        for (r, m) in st.session_state.messages
                    ][-10:]
                    msg_history.append({"role":"assistant", "content": state_summary_for_llm()})
                    nxt = llm_chat(INTAKE_SYSTEM_PROMPT, msg_history, model="gpt-4o-mini", max_tokens=80)

                    lower = nxt.lower()
                    field = None
                    if "task" in lower and "affected" in lower: field = "task_ctx.title"
                    elif "due" in lower or "deadline" in lower: field = "task_ctx.deadline_text"
                    elif "minute" in lower or "how long" in lower: field = "task_ctx.est_minutes"
                    elif "feel" in lower or "emotion" in lower: field = "feelings_words"
                    elif "sleep" in lower: field = "sleep_hours"
                    elif "energy" in lower: field = "energy_0_3"
                    elif any(k in lower for k in ["perfectionism","fear of failure","low interest","overwhelm","distraction"]):
                        field = "non_disorder_causes_selected"

                    ask(nxt, field=field)

                except Exception:
                    path, q = next_missing_question()
                    ask(q if q else "Tell me a bit more about what makes starting hard.", field=path)
