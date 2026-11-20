import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

# Starter FAQ knowledge base. Edit or add questions and answers for your chosen topic.
FAQS = [
    {
        "question": "What is this internship?",
        "answer": "This is an AI internship project where you build practical applications and learn by implementing them."
    },
    {
        "question": "Which programming language are we using?",
        "answer": "We are using Python because it has useful libraries for AI, NLP, and building applications."
    },
    {
        "question": "What is Python?",
        "answer": "Python is a beginner-friendly, high-level programming language used for web development, automation, data science, and artificial intelligence."
    },
    {
        "question": "What is artificial intelligence?",
        "answer": "Artificial Intelligence (AI) is a field of computer science focused on building systems that perform tasks that normally require human-like intelligence."
    },
    {
        "question": "What is NLP?",
        "answer": "Natural Language Processing (NLP) helps computers process, understand, and work with human language."
    },
    {
        "question": "What is a chatbot?",
        "answer": "A chatbot is a software application that interacts with users through text or voice and provides automated responses."
    },
    {
        "question": "How do I run this project?",
        "answer": "Open a terminal in the FAQ_Chatbot folder, install the packages with pip install -r requirements.txt, then run streamlit run app.py."
    },
    {
        "question": "How does this chatbot find answers?",
        "answer": "It converts FAQ questions into TF-IDF vectors and uses cosine similarity to find the question most similar to the user's message."
    },
    {
        "question": "What is TF-IDF?",
        "answer": "TF-IDF is a numerical method that gives words weights based on how important they are in a document compared with a collection of documents."
    },
    {
        "question": "What is cosine similarity?",
        "answer": "Cosine similarity measures how similar two text vectors are by comparing the angle between them. A higher score means greater similarity."
    },
    {
        "question": "How can I contact support?",
        "answer": "Add your organization's verified support email or contact details to the FAQ knowledge base before publishing this chatbot."
    },
]

def clean_text(text):
    """Lowercase text and remove punctuation and extra spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def find_answer(user_question, threshold=0.18):
    """Return the best matching FAQ answer and its similarity score."""
    cleaned_faqs = [clean_text(item["question"]) for item in FAQS]
    cleaned_user = clean_text(user_question)
    if not cleaned_user:
        return None, 0.0, None

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(cleaned_faqs + [cleaned_user])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])

    if best_score < threshold:
        return None, best_score, None
    return FAQS[best_index]["answer"], best_score, FAQS[best_index]["question"]

st.title("💬 FAQ Chatbot")
st.write("Ask a question about Python, AI, NLP, or this starter internship project.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me a question about Python, AI, NLP, or this project."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Type your question...")
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    answer, score, matched_question = find_answer(user_question)

    if answer:
        reply = answer + f"\n\n*Matched FAQ: “{matched_question}” (similarity: {score:.2f})*"
    else:
        reply = "Sorry, I couldn't find a close answer in my FAQ list. Try asking in a different way or add this question to the FAQ knowledge base."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

with st.sidebar:
    st.header("About this chatbot")
    st.write("This chatbot uses NLP preprocessing, TF-IDF, and cosine similarity.")
    st.write("The FAQ list is a small starter knowledge base that you can customize.")
    if st.button("Clear chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. What would you like to ask?"}
        ]
        st.rerun()
