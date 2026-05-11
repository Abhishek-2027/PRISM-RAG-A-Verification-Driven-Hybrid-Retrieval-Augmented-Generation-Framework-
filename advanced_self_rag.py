import os
import json
import time
import warnings
from typing import List, Optional, Literal
from typing_extensions import TypedDict
import numpy as np
from dotenv import load_dotenv

# Filter warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Imports for LangChain and LangGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from langgraph.graph import END, StateGraph, START

# PART 1: Hybrid Retrieval imports
from rank_bm25 import BM25Okapi
import faiss
from langchain_community.vectorstores import FAISS

print("--- INITIALIZING ADVANCED SELF-RAG ---")

# 1. Embeddings & LLM (Using Llama 3.3 70B on Groq)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama-3.3-70b-versatile")
verifier_llm_engine = ChatGoogleGenerativeAI(model="gemini-2.5-flash-native-audio-latest")

# 2. Vector Store Setup
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

print("Loading documents from web...")
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

print(f"Indexing {len(doc_splits)} chunks into Chroma...")
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. Hybrid Retrieval Implementation
class BM25Index:
    def __init__(self, documents: list):
        self.documents = documents
        tokenised = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenised)
        print(f"BM25Index built over {len(documents)} chunks.")

    def search(self, query: str, k: int = 10) -> list:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(scores[i], self.documents[i]) for i in top_indices if scores[i] > 0]

def reciprocal_rank_fusion(ranked_lists: list[list], k_rrf: int = 60) -> list:
    from collections import defaultdict
    rrf_scores = defaultdict(float)
    doc_map = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content[:200]
            rrf_scores[key] += 1.0 / (rank + k_rrf)
            doc_map[key] = doc
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys]

class HybridRetriever:
    def __init__(self, bm25_index, dense_retriever, k_sparse=10, k_dense=10, k_final=5):
        self.bm25_index = bm25_index
        self.dense_retriever = dense_retriever
        self.k_sparse = k_sparse
        self.k_dense = k_dense
        self.k_final = k_final

    def invoke(self, query: str) -> list:
        ranked_sparse = [doc for _s, doc in self.bm25_index.search(query, k=self.k_sparse)]
        ranked_dense = self.dense_retriever.invoke(query)
        fused = reciprocal_rank_fusion([ranked_sparse, ranked_dense])
        return fused[:self.k_final]

bm25_index = BM25Index(doc_splits)
hybrid_retriever = HybridRetriever(bm25_index, retriever)

# 4. Component Definitions (LLM Chains)

# Uncertainty Controller
class ControlDecision(BaseModel):
    decision: Literal["answer_directly", "retrieve", "abstain"]
    confidence: float
    reasoning: str

controller_llm = llm.with_structured_output(ControlDecision)
controller_prompt = ChatPromptTemplate.from_messages([
    ("system", "Decide if question needs retrieval (about agents/prompt engineering), can be answered directly (general), or abstain (unrelated)."),
    ("human", "User question: {question}")
])
uncertainty_controller = controller_prompt | controller_llm

# Reasoning Planner
class ReasoningPlan(BaseModel):
    is_simple: bool
    sub_queries: List[str]
    reasoning_strategy: str

planner_llm = llm.with_structured_output(ReasoningPlan)
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "Decompose complex questions into 2-4 sub-queries for retrieval. For simple ones, return 1 sub-query."),
    ("human", "Question: {question}")
])
reasoning_planner = planner_prompt | planner_llm

# Verifier
class VerificationResult(BaseModel):
    factual_grounding: Literal["supported", "partial", "unsupported"] = Field(description="Is the answer grounded in evidence?")
    answers_question: bool = Field(description="Does it answer the original question?")
    verdict: Literal["pass", "refine"] = Field(description="Pass or need refinement?")
    evidence_gaps: str = Field(description="List information gaps as a comma-separated string, or 'none' if empty")

verifier_prompt = ChatPromptTemplate.from_messages([
    ("system", "Verify if answer is grounded in evidence. Decide 'pass' or 'refine'."),
    ("human", "Question: {question}\nEvidence: {evidence}\nAnswer: {answer}")
])

external_verifier = verifier_prompt | verifier_llm_engine.with_structured_output(VerificationResult)
llama_verifier = verifier_prompt | llm.with_structured_output(VerificationResult)

# Generator & Refiner
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | llm

refinement_prompt = ChatPromptTemplate.from_messages([
    ("system", "Refine the draft answer using new evidence for identified gaps."),
    ("human", "Question: {question}\nDraft: {draft_answer}\nGaps: {gaps}\nNew Evidence: {additional_evidence}")
])
refiner_chain = refinement_prompt | llm | StrOutputParser()

# 5. Graph State & Nodes

class AdvancedAgentState(TypedDict):
    question: str
    documents: List
    generation: str
    control_decision: str
    sub_queries: List[str]
    verification_verdict: str
    evidence_gaps: List[str]
    refinement_done: bool

def run_uncertainty_controller(state):
    print("\n[1] UNCERTAINTY CONTROLLER")
    res = uncertainty_controller.invoke({"question": state["question"]})
    print(f"Decision: {res.decision}")
    return {"control_decision": res.decision, "refinement_done": False}

def run_reasoning_planner(state):
    print("\n[2] REASONING PLANNER")
    res = reasoning_planner.invoke({"question": state["question"]})
    print(f"Plan: {res.sub_queries}")
    return {"sub_queries": res.sub_queries}

def run_multi_hop_retrieval(state):
    print("\n[3] MULTI-HOP RETRIEVAL")
    all_docs = []
    seen = set()
    for q in state["sub_queries"]:
        docs = hybrid_retriever.invoke(q)
        for d in docs:
            h = hash(d.page_content[:200])
            if h not in seen:
                seen.add(h)
                all_docs.append(d)
    print(f"Retrieved {len(all_docs)} unique documents.")
    return {"documents": all_docs}

def run_generator(state):
    print("\n[4] GENERATOR")
    res = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"generation": res.content}

def run_external_verifier(state):
    print("\n[5] EXTERNAL VERIFIER")
    evidence = "\n".join([d.page_content for d in state["documents"][:5]])
    try:
        res = external_verifier.invoke({
            "question": state["question"],
            "evidence": evidence,
            "answer": state["generation"]
        })
    except Exception as e:
        print(f"Gemini verification failed ({e}), falling back to Llama.")
        res = llama_verifier.invoke({
            "question": state["question"],
            "evidence": evidence,
            "answer": state["generation"]
        })
    
    print(f"Verdict: {res.verdict}")
    gaps = [g.strip() for g in res.evidence_gaps.split(",") if g.strip().lower() != "none"]
    return {"verification_verdict": res.verdict, "evidence_gaps": gaps}

def run_bounded_refiner(state):
    print("\n[6] BOUNDED REFINER")
    new_docs = []
    for gap in state["evidence_gaps"]:
        new_docs.extend(hybrid_retriever.invoke(gap)[:2])
    
    refined = refiner_chain.invoke({
        "question": state["question"],
        "draft_answer": state["generation"],
        "gaps": ", ".join(state["evidence_gaps"]),
        "additional_evidence": "\n".join([d.page_content for d in new_docs])
    })
    return {"generation": refined, "documents": state["documents"] + new_docs, "refinement_done": True}

def run_direct_answer(state):
    print("\n[DIRECT] Generating answer directly...")
    res = llm.invoke(state["question"])
    return {"generation": res.content, "refinement_done": True}

def run_abstain(state):
    print("\n[ABSTAIN] Question out of domain.")
    return {"generation": "I'm sorry, I can only answer questions about AI agents and prompt engineering.", "refinement_done": True}

# 6. Build Graph

workflow = StateGraph(AdvancedAgentState)

workflow.add_node("controller", run_uncertainty_controller)
workflow.add_node("planner", run_reasoning_planner)
workflow.add_node("retriever", run_multi_hop_retrieval)
workflow.add_node("generator", run_generator)
workflow.add_node("verifier", run_external_verifier)
workflow.add_node("refiner", run_bounded_refiner)
workflow.add_node("direct", run_direct_answer)
workflow.add_node("abstain", run_abstain)

workflow.add_edge(START, "controller")

def route_controller(state):
    if state["control_decision"] == "answer_directly": return "direct"
    if state["control_decision"] == "abstain": return "abstain"
    return "planner"

workflow.add_conditional_edges("controller", route_controller, {"direct": "direct", "abstain": "abstain", "planner": "planner"})
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", "verifier")

def route_verifier(state):
    if state["refinement_done"] or state["verification_verdict"] == "pass": return "end"
    return "refiner"

workflow.add_conditional_edges("verifier", route_verifier, {"end": END, "refiner": "refiner"})
workflow.add_edge("refiner", "verifier")
workflow.add_edge("direct", END)
workflow.add_edge("abstain", END)

app = workflow.compile()

# 7. Test Execution Suite for Research Paper
if __name__ == "__main__":
    test_cases = [
        {"type": "IN-CONTEXT", "query": "Explain the ReAct framework and how it combines reasoning and acting."},
        {"type": "IN-CONTEXT", "query": "How do autonomous agents use Vector Databases for long-term memory?"},
        {"type": "IN-CONTEXT", "query": "What are the limitations of Chain-of-Thought prompting in complex reasoning?"},
        {"type": "IN-CONTEXT", "query": "Compare 'Tree of Thoughts' and 'Chain of Thought' prompting architectures."},
        {"type": "GENERAL (DIRECT)", "query": "What is the speed of light in a vacuum?"},
        {"type": "GENERAL (DIRECT)", "query": "What is the chemical symbol for Gold?"},
        {"type": "GENERAL (DIRECT)", "query": "Who is the author of 'The Great Gatsby'?"},
        {"type": "OUT-OF-DOMAIN", "query": "What is the best way to train a golden retriever puppy?"},
        {"type": "OUT-OF-DOMAIN", "query": "How do you calculate the area of a circle?"},
        {"type": "OUT-OF-DOMAIN", "query": "Tell me a brief history of the Apollo 11 moon landing."}
    ]
    
    results_file = "rag_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("ADVANCED SELF-RAG PERFORMANCE EVALUATION\n")
        f.write("==================================================\n\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- RUNNING TEST {i}/{len(test_cases)}: {case['type']} ---")
        print(f"Question: {case['query']}")
        
        start_time = time.time()
        final_state = app.invoke({"question": case['query']})
        latency = time.time() - start_time
        
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"TEST CASE {i}: {case['type']}\n")
            f.write(f"QUESTION: {case['query']}\n")
            f.write(f"DECISION: {final_state.get('control_decision', 'N/A')}\n")
            f.write(f"LATENCY: {latency:.2f} seconds\n")
            f.write(f"VERDICT: {final_state.get('verification_verdict', 'N/A')}\n")
            f.write(f"REFINEMENT DONE: {final_state.get('refinement_done', False)}\n")
            f.write("-" * 30 + "\n")
            f.write("FINAL ANSWER:\n")
            f.write(f"{final_state['generation']}\n")
            f.write("=" * 50 + "\n\n")
            
        print(f"Test {i} complete. Latency: {latency:.2f}s")

    print(f"\nAll tests finished. Results saved to {results_file}")
