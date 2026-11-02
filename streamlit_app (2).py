import re
import time
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="AI Admission Guide Chatbot", page_icon="🎓")

# Key and model come from Streamlit Secrets (never written in the code)
API_KEY = st.secrets.get("GEMINI_API_KEY")
MODEL = st.secrets.get("GEMINI_MODEL", "gemini-flash-latest")
# Backup models are tried automatically if the main model fails
FALLBACKS = [m.strip() for m in str(st.secrets.get("GEMINI_FALLBACKS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite")).split(",") if m.strip()]
MODELS = [MODEL] + [m for m in FALLBACKS if m != MODEL]

SYSTEM_PROMPT = """You are 'AI Admission Guide', a friendly assistant that helps students
and parents in India understand admission processes, mainly engineering (MHT-CET, JEE, CAP rounds)
and also diploma, pharmacy, MBA, and other courses.
Help with: eligibility and percentage criteria, required documents, fees and scholarships,
admission steps and timelines, branch selection (e.g. ENTC, Computer, Mechanical, Civil),
and career options after engineering.
Rules:
- Answer in simple, short, step-by-step language. Use bullet points where useful.
- If the student writes in Hindi or Marathi, reply in the same language.
- Fees, cut-offs and dates change every year, so give general guidance and
  always advise checking the official website or the college admission office.
- Never invent exact numbers, dates or links. If unsure, say so.
- If the question is not about education or admissions, politely say you can only help with admissions."""

st.title("🎓 AI Admission Guide Chatbot")
st.caption("Ask about engineering and other admissions: eligibility, percentage, documents, fees, process, branches.")

if not API_KEY:
    st.error("Setup error: GEMINI_API_KEY is not set in the app secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)


def short_error(e):
    """Keep only the error code and status, e.g. '503 UNAVAILABLE'."""
    m = re.search(r"\b(\d{3})\s+([A-Z_]+)", str(e))
    return f"{m.group(1)} {m.group(2)}" if m else str(e)[:80]


def ask_gemini(contents):
    """Try each model in turn; retry once on temporary overload (503) or rate limit (429)."""
    errors = []
    for model in MODELS:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.4,
                    ),
                )
                if resp.text:
                    return resp.text
                errors.append(f"{model}: empty response")
                break
            except Exception as e:
                msg = str(e)
                if attempt == 0 and ("503" in msg or "429" in msg):
                    time.sleep(3)
                    continue
                errors.append(f"{model}: {short_error(e)}")
                break
    raise RuntimeError("; ".join(errors))


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Try asking")
    st.write("- What documents are needed for engineering admission?")
    st.write("- What is the CAP round process?")
    st.write("- Which branch is good if I like electronics?")
    st.write("- Career options after ENTC engineering?")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Type your admission question here...")

if question is not None:
    if not question.strip():
        st.warning("Please type an admission-related question.")
    else:
        # error messages are shown but not sent back to the model
        history = [m for m in st.session_state.messages if not m.get("error")]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["content"])],
            )
            for m in history + [{"role": "user", "content": question}]
        ]

        with st.chat_message("assistant"):
            failed = False
            try:
                with st.spinner("Thinking..."):
                    answer = ask_gemini(contents)
            except Exception as e:
                failed = True
                answer = ("Sorry, I could not get an answer right now. Please try again in a minute.\n\n"
                          f"Technical details: `{str(e)[:300]}`")
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "error": failed})
