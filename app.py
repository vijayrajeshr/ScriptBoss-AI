import os
import numpy as np
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langchain_openrouter import ChatOpenRouter

load_dotenv()

# --- 1. DEFINING THE STATE (The Clipboard) ---
class GraphState(TypedDict):
    document_text: str       
    query: str               
    retrieved_context: str   
    attempts: int            
    evaluation: str          
    response: str            

# --- 2. VERCEL-FRIENDLY LIGHTWEIGHT SEARCH ENGINE ---
def compute_cosine_similarity(vec1, vec2):
    """Calculates semantic matching scores without needing heavy 500MB ML libraries."""
    all_words = list(set(list(vec1.keys()) + list(vec2.keys())))
    v1 = [vec1.get(w, 0) for w in all_words]
    v2 = [vec2.get(w, 0) for w in all_words]
    dot_prod = np.dot(v1, v2)
    norm_a = np.linalg.norm(v1)
    norm_b = np.linalg.norm(v2)
    if not norm_a or not norm_b:
        return 0.0
    return float(dot_prod / (norm_a * norm_b))

# --- 3. THE GRAPH NODES (Work Stations) ---
def retrieve_node(state: GraphState):
    """Chunks the text document and pulls the best semantic match."""
    print("🤖 [Node: Retriever] Scanning document for information...")
    text = state["document_text"]
    query = state["query"]
    
    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 20]
    if not chunks:
        return {"retrieved_context": "No valid data found.", "attempts": state.get("attempts", 0) + 1}
        
    query_freq = {w: query.lower().count(w) for w in query.lower().split()}
    best_chunk = chunks[0]
    highest_score = -1.0
    
    for chunk in chunks:
        chunk_freq = {w: chunk.lower().count(w) for w in chunk.lower().split()}
        score = compute_cosine_similarity(query_freq, chunk_freq)
        if score > highest_score:
            highest_score = score
            best_chunk = chunk
            
    return {"retrieved_context": best_chunk, "attempts": state.get("attempts", 0) + 1}

def grade_context_node(state: GraphState):
    """The Auditor: Checks if the pulled data actually answers the prompt."""
    print("🤖 [Node: Grader] Auditing data relevance...")
    llm = ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0.0)
    
    grader_prompt = """
    Analyze the following retrieved context and decide if it contains enough information to answer the user query.
    
    User Query: {query}
    Retrieved Context: {context}
    
    Respond with exactly one word: 'good' if it answers the question, or 'bad' if it is missing core details.
    """
    prompt = PromptTemplate.from_template(grader_prompt).format(
        query=state["query"], context=state["retrieved_context"]
    )
    
    result = llm.invoke(prompt).content.strip().lower()
    evaluation = "good" if "good" in result else "bad"
    return {"evaluation": evaluation}

def query_rewrite_node(state: GraphState):
    """The Optimizer: Rewrites the question to search better if the first pass failed."""
    print("⚠️ [Node: Rewriter] Search failed to find good data. Optimizing query keywords...")
    llm = ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0.5)
    
    rewrite_prompt = """
    The search query '{query}' failed to extract accurate documentation context. 
    Provide an optimized, broader version of this query focusing on generalized synonyms or core technical concepts.
    Respond with only the updated query text, nothing else.
    """
    prompt = PromptTemplate.from_template(rewrite_prompt).format(query=state["query"])
    new_query = llm.invoke(prompt).content.strip()
    return {"query": new_query}

def generate_podcast_node(state: GraphState):
    """The Creative Writer: Turns raw technical data into a podcast script."""
    print("🎨 [Node: Writer] Crafting a conversational script...")
    llm = ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0.7)
    
    script_prompt = """
    You are a podcast writer. Create an entertaining, fast-paced dialogue between two hosts, Alex and Sam. 
    They are breaking down the following source material for a non-technical audience.
    
    Source Material: {context}
    Target Subject: {query}
    
    Make it witty and format it exactly like:
    Alex: [Dialogue]
    Sam: [Dialogue]
    Keep it within 4-6 lines total.
    """
    prompt = PromptTemplate.from_template(script_prompt).format(
        context=state["retrieved_context"], query=state["query"]
    )
    response = llm.invoke(prompt).content
    return {"response": response}

# --- 4. THE INTERSECTION (The Traffic Cop Logic) ---
def route_evaluation(state: GraphState):
    if state["evaluation"] == "good" or state["attempts"] >= 2:
        if state["attempts"] >= 2 and state["evaluation"] == "bad":
            print("🛑 Force-routing to generator to prevent infinite looping.")
        return "generate"
    return "rewrite"

# --- 5. ASSEMBLING THE GRAPH ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_context_node)
workflow.add_node("rewrite", query_rewrite_node)
workflow.add_node("generate", generate_podcast_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    route_evaluation,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)
workflow.add_edge("rewrite", "retrieve") 
workflow.add_edge("generate", END)

app = workflow.compile()

#docker-workflow test 1 : cancelled
#docker-workflow test 2 : 