import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"  # Change if your backend runs elsewhere

st.title("Research Paper Summarizer & Q&A")

# State management for session_id and summaries
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "summaries" not in st.session_state:
    st.session_state["summaries"] = None
if "last_uploaded_file_name" not in st.session_state:
    st.session_state["last_uploaded_file_name"] = None


# Upload PDF
st.header("Upload Your Research Paper")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if uploaded_file.name != st.session_state["last_uploaded_file_name"]:
        with st.spinner("Processing PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        with st.spinner("Generating Summary..."):
            response = requests.post(f"{BACKEND_URL}/upload/", files=files)
            if response.status_code == 200:
                data = response.json()
                st.session_state["session_id"] = data["session_id"]
                st.session_state["summaries"] = data["summaries"]
                st.session_state["last_uploaded_file_name"] = uploaded_file.name
                st.success("Summary generated!")
            else:
                st.error("Error uploading or processing PDF.")

# Display section-wise summaries in preferred order
section_order = [
    "Abstract", "Introduction", "Background", "Related Work",
    "Methods", "Methodology", "Experiment", "Results", "Discussion",
    "Conclusion", "References", "Acknowledgments"
]

if st.session_state["summaries"]:
    st.header("Section-wise Summaries")
    summaries = st.session_state["summaries"]
    # Show ordered sections first
    for section in section_order:
        if section in summaries:
            st.markdown(f"### {section}")
            st.write(summaries[section])
    # Show any remaining sections
    for section in summaries:
        if section not in section_order:
            st.markdown(f"### {section}")
            st.write(summaries[section])

# Q&A Section
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "qa_input" not in st.session_state:
    st.session_state["qa_input"] = ""

if st.session_state["session_id"]:
    st.header("Q&A about the Paper")

    # Display chat history
    for q, a in st.session_state["chat_history"]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Answer:** {a}")

    st.subheader("Ask Questions About the Paper")

    # Chat input using a form
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("Type your question and press Enter", key="qa_input")
        submitted = st.form_submit_button("Ask")
        if submitted and query:
            with st.spinner("Searching and generating answer..."):
                response = requests.post(
                    f"{BACKEND_URL}/ask/",
                    data={"session_id": st.session_state["session_id"], "query": query},
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(f"**You:** {query}")
                    st.markdown(f"**Answer:** {answer}")
                    # Append to chat history
                    st.session_state["chat_history"].append((query, answer))
                    # Clear the input box (will clear on next render due to clear_on_submit)
                else:
                    st.error("Failed to retrieve answer from backend.")
