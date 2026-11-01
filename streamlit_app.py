import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="AI Admission Guide Chatbot", page_icon="🎓")

# Key and model come from Streamlit Secrets (never written in the code)
API_KEY = st.secrets.get("GEMINI_API_KEY")
MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")

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
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.4,
                            max_output_tokens=800,
                        ),
                    )
                answer = resp.text or "I could not generate an answer. Please rephrase your question."
            except Exception:
                answer = ("Sorry, the service is busy or the daily free limit is reached. "
                          "Please wait a minute and try again.")
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
