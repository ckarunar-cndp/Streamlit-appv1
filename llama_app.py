import os
import streamlit as st
from PIL import Image
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import openai

# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="🖤 CNDP Consulting AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"  # sidebar closed by default for Squarespace embed
) 

# ---------------- Custom Dark Theme ----------------
st.markdown(
    """
    <style>
    body, .stApp, .css-18e3th9 {background-color: #000000; color: #e2e6f3;}
    [data-testid="stSidebar"] {
        background-color: #0e0f11 !important;
        color: #e2e6f3;
        min-width: 260px;
        max-width: 260px;
    }
    [data-testid="stSidebarNav"] {display: none;}
    button, .stTextInput>div>div>input {
        background-color: #2a2b33; color: #e2e6f3; border-radius: 6px;
    }
    button:hover {background-color: #3b3c46;}
    h1, h2, h3 {color: #e2e6f3;}
    .user-bubble, .assistant-bubble {
        border-radius: 12px; padding: 10px; margin: 8px 0;
    }
    .user-bubble {background-color: #2a2b33;}
    .assistant-bubble {background-color: #1f1f25;}

    /* Sidebar Styles */
    .sidebar-title {
        text-align: center;
        color: #E2E6F3;
        font-size: 20px;
        font-weight: 600;
        margin-top: 10px;
    }
    .online-dot {
        height: 10px;
        width: 10px;
        background-color: #1DB954;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #1DB954;
        margin-top: 5px;
    }

    /* Chat Section Header */
    .center-header {
        text-align: center;
        margin-top: -20px;
        margin-bottom: 10px;
    }
    .center-header h1 {
        color: #E2E6F3;
        font-size: 28px;
        margin-bottom: 0px;
    }
    .center-header p {
        color: #AAAAAA;
        font-size: 16px;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Sidebar ----------------
logo_path = "ML Logo 1.png"

with st.sidebar:
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, use_column_width=True)
        st.markdown("<div style='text-align:center'><span class='online-dot'></span></div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Logo not found. Please ensure 'ML Logo 1.png' is in your app folder.")

# ---------------- Chat Section Header ----------------
st.markdown(
    """
    <div class='center-header'>
        <h1>🤖 CNDP Consulting AI Assistant</h1>
        <p>AI-powered knowledge companion ✨</p>
    </div>
    """,
    unsafe_allow_html=True
)



# ---------------- API Key Setup ----------------
openai.api_key = os.getenv("OPENAI_API_KEY") or st.text_input("🔑 Enter your OpenAI API Key", type="password")

# ---------------- Session Management ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome! How may I assist you today?"}
    ]

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

# ---------------- Sidebar Info ----------------
with st.sidebar:
    st.markdown("### 🏢 CNDP Consulting Inc.")
    st.markdown("💡 Tagline: Inspire. Build. Transform")
    st.markdown("🌐 [Visit Our Website](https://www.cndpconsultinginc.com/)")
   # st.markdown("📧 info@cndpconsult.com")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/company/cndp-consulting-inc/about/?viewAsMember=true)")
    st.markdown("📍 Milton, Ontario, Canada")
    st.markdown("📧Open a support ticket: (info@cndpconsult.com)")
    st.markdown("---")

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chat cleared. How may I help you today?"}]
    st.button("🧹 Clear Chat", on_click=clear_chat_history)
# ---------------- Load Knowledge Base ----------------
@st.cache_resource(show_spinner=False)
def load_index():
    with st.spinner("⚙️ Loading CNDP knowledge base... please wait."):
        reader = SimpleDirectoryReader(input_dir="./documents", recursive=True)
        docs = reader.load_data()

        Settings.llm = OpenAI(model="gpt-4-turbo")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

        return VectorStoreIndex.from_documents(docs)

if st.session_state.chat_engine is None:
    index = load_index()
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=False)

# ---------------- FAQ Section ----------------
st.markdown("---")
st.markdown("<h3 style='text-align:center;'>💬 Frequently Asked Questions</h3>", unsafe_allow_html=True)

faq_list = {
    "💼 What services does CNDP Consulting provide?": 
        "We offer AI, data analytics, and cloud consulting solutions tailored for modern businesses.",
    "🏢 Which industries has CNDP Consulting served?": 
        "Our expertise spans Travel & Tourism, Legal, Tax, Construction, Financial Services, and SMBs adopting AI solutions.",
    "💪 What are CNDP’s core strengths?": 
        "AI-driven insights, predictive modeling, automation, and scalable architecture.",
    "☁️ Does CNDP provide cloud-based solutions?": 
        "Yes! We specialize in GCP, AWS, and Azure integrations.",
}

for question, answer in faq_list.items():
    with st.expander(question, expanded=False):
        st.markdown(f"<p style='color:#EAEAEA;'>{answer}</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Chat Interface ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>🙋‍♀️ {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-bubble'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

# Input field (AFTER showing chat to ensure correct rendering)
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🤔 Thinking..."):
        try:
            response = st.session_state.chat_engine.chat(prompt)
            answer_text = response.response.strip()

            if not answer_text:
                answer_text = (
                    "I'm sorry, I couldn’t find the exact details. "
                    "To learn more about our pricing or project rates, please contact us directly via "
                    "<a href='https://www.cndpconsultinginc.com/contact' target='_blank'>our Contact Page</a> "
                    "or email <b>info@cndpconsult.com</b>."
                )

        except Exception as e:
            answer_text = (
                f"⚠️ Sorry, there was an issue generating the response. ({str(e)}) "
                "Please reach out to us via email at info@cndpconsult.com."
            )

    # Append response and rerun safely
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    st.rerun()
