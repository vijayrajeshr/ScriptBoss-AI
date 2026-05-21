# ScriptBoss AI 

<img width="1919" height="922" alt="image" src="https://github.com/user-attachments/assets/c00d2d3c-80ba-46ec-9489-d53bf9b4c1d1" />


<img width="812" height="283" alt="image" src="https://github.com/user-attachments/assets/0b4e8950-4dc4-4753-a393-68c37040b3e6" />

<img width="936" height="483" alt="image" src="https://github.com/user-attachments/assets/478fd699-aee9-43ff-89d8-e7f0b52ff9ad" />

<img width="1158" height="684" alt="image" src="https://github.com/user-attachments/assets/3cd2e27d-7641-4e35-a9a2-4791e309b2fe" />

<img width="1696" height="755" alt="image" src="https://github.com/user-attachments/assets/a1ec7415-a223-4e54-8f6d-f13ed737a617" />





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
