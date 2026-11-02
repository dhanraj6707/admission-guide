import time
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="AI Admission Guide Chatbot", page_icon="🎓")

# Key and model come from Streamlit Secrets (never written in the code)
API_KEY = st.secrets.get("GEMINI_API_KEY")
MODEL = st.secrets.get("GEMINI_MODEL", "gemini-flash-latest")
# Backup models are tried automatically if the main model is overloaded (503) or unavailable
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

client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=25000, retry_options=types.HttpRetryOptions(attempts=1)))


def ask_gemini(contents):
    """Try each model; retry once on temporary errors such as 503 high demand."""
    last_err = None
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
                last_err = Exception("Empty response from model")
            except Exception as e:
                last_err = e
                msg = str(e)
                if "404" in msg or "NOT_FOUND" in msg:
                    break  # model name not available, go to the next model
                if attempt == 0:
                    time.sleep(2)
    raise last_err


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
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["content"])],
            )
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer = ask_gemini(contents)
            except Exception as e:
                answer = ("Sorry, I could not get an answer right now. Please try again in a minute.\n\n"
                          f"Technical details: `{str(e)[:300]}`")
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
