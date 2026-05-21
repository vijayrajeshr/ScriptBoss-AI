import streamlit as st
from app import app 

st.set_page_config(page_title="ScriptBoss AI", page_icon="🎙️", layout="centered")

st.title("🎙️ ScriptBoss AI")
st.subheader("Autonomous, Self-Correcting Document-to-Podcast Generator")
st.write("Upload text, define a topic, and watch a multi-agent LangGraph RAG pipeline self-correct to generate an audio script.")

st.divider()

source_text = st.text_area(
    "📄 Paste Source Documentation Text Here:",
    placeholder="Paste chapters, document sections, or technical data here...",
    height=200
)

user_query = st.text_input(
    "🎯 What specific topic should the podcast hosts talk about?",
    placeholder="e.g., How do StateGraph loops work?"
)

if st.button("🚀 Run ScriptBoss AI Engine", use_container_width=True):
    if not source_text or not user_query:
        st.error("Please provide both source text and a target query topic!")
    else:
        with st.spinner("ScriptBoss AI is executing the pipeline..."):
            initial_state = {
                "document_text": source_text,
                "query": user_query,
                "attempts": 0,
                "retrieved_context": "",
                "evaluation": "",
                "response": ""
            }
            final_output = app.invoke(initial_state)
            
        st.success("Pipeline Execution Complete!")
        st.divider()
        
        st.write("### 📻 Generated Podcast Transcript")
        st.text_area("Final Script Output:", value=final_output["response"], height=250)
        
        with st.expander("🧠 Peek Inside the ScriptBoss Architecture"):
            st.write(f"**Final Search Query Used:** `{final_output['query']}`")
            st.write(f"**Total Loop Attempts Needed:** `{final_output['attempts']}`")
            st.write(f"**Final Audit Evaluation:** `{final_output['evaluation'].upper()}`")
            st.write(f"**Extracted RAG Context Chunk:**")
            st.info(final_output["retrieved_context"])

st.caption("Powered by LangChain + LangGraph • 100% Serverless & Free-Tier Optimized")