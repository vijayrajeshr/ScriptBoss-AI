# 🎙️ ScriptBoss AI

**ScriptBoss AI** is an autonomous, serverless-ready multi-agent RAG pipeline designed to ingest complex technical documents and synthesize them into highly engaging, conversational two-person podcast scripts.

Built using the core orchestration capabilities of **LangGraph** and the standard abstraction toolkits of **LangChain**, this project implements an advanced structural AI pattern: **Corrective Retrieval-Augmented Generation (CRAG)** featuring an evaluation and optimization loop.

---

## 🚀 Key Architectural Highlights (For Interviewers)

Most portfolio RAG applications execute a basic, linear pipeline ($Retrieval \rightarrow Generation$) which breaks down when inaccurate or noisy document chunks are passed to the LLM. **ScriptBoss AI** introduces structural self-correction:

*   **Stateful Memory Architecture:** Uses a centralized `GraphState` schema to elegantly manage state mutations across disconnected computational nodes.
*   **The Auditor Loop (Self-Correction):** Features a dedicated *Grader Node* that critically audits retrieved context relevance before allowing generation.
*   **Dynamic Query Rewriting:** If context verification fails, data is automatically rerouted into a *Query Rewrite Node* which optimizes terms to query the document successfully on a secondary pass.
*   **Serverless-Optimized Design:** Built completely free of heavy native C-compiled machine learning frameworks (`faiss-cpu`, `torch`). It introduces an inline, high-performance vector semantic parser running safely within Vercel's strict **50MB runtime zip limit**.

---

## 🧠 The LangGraph Decision Workflow

```text
       [ User Ingests Text & Defines Subject ]
                         │
                         ▼
               ┌───────────────────┐
               │  Retriever Node   │ ◄───┐
               └───────────────────┘     │
                         │               │ (Rerouted Loop)
                         ▼               │
               ┌───────────────────┐     │
               │    Grader Node    │     │
               └───────────────────┘     │
                         │               │
                         ▼               │
        🔀 [ Conditional Edge: Evaluation ]   │
            ├──► BAD  ──► [ Query Rewrite Node ]
            └──► GOOD ──► [ Writer Node ] ──► [ END ]