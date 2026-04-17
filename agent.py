import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
import chromadb
from sentence_transformers import SentenceTransformer
from importlib.metadata import version

# Load environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY", "")
if not groq_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Knowledge Base Documents
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "OPD Timings",
        "text": "MediCare General Hospital operates Outpatient Department (OPD) services from Monday to Saturday. The general OPD timings are from 9:00 AM to 5:00 PM. Some departments such as cardiology and orthopedics may have extended hours depending on doctor availability. Sunday OPD is limited and only emergency consultations are handled. Patients are advised to arrive at least 30 minutes early for registration. Token systems are used to manage patient flow. Online appointment holders are given priority over walk-in patients. Changes in OPD schedules during holidays or special circumstances are updated on the hospital website and helpline."
    },
    {
        "id": "doc_002",
        "topic": "Doctor Consultation and Departments",
        "text": "MediCare General Hospital provides multi-specialty care including cardiology, orthopedics, neurology, pediatrics, dermatology, and general medicine. Patients should choose departments based on symptoms. For chest pain or heart-related issues, cardiology is recommended. Bone injuries or joint pain are handled by orthopedics. Skin-related concerns fall under dermatology. Pediatricians are available for child healthcare. General physicians handle common illnesses such as fever, infections, and routine checkups. The hospital reception or assistant system can help guide patients to the appropriate department based on their symptoms."
    },
    {
        "id": "doc_003",
        "topic": "Consultation Fees",
        "text": "Consultation fees at MediCare General Hospital vary depending on the department and doctor. General physician consultation typically costs between ₹300 to ₹500. Specialist consultations such as cardiology, neurology, and orthopedics range from ₹600 to ₹1200. Follow-up consultations within 7 days may be discounted or free depending on hospital policy. Emergency consultation charges are higher and may include additional service fees. Payment can be made via cash, card, or digital payment methods. Patients are advised to confirm exact fees at the reception before booking."
    },
    {
        "id": "doc_004",
        "topic": "Insurance Coverage",
        "text": "MediCare General Hospital accepts a wide range of health insurance providers including Star Health, Arogya, ICICI Lombard, and HDFC ERGO. Cashless treatment is available for admitted patients subject to policy approval. Outpatient consultations may or may not be covered depending on the insurance plan. Patients must carry valid ID proof, insurance card, and pre-authorization documents if required. The insurance desk at the hospital assists patients with claim processing, approvals, and documentation. It is recommended to verify coverage details with both the hospital and insurance provider before treatment."
    },
    {
        "id": "doc_005",
        "topic": "Appointment Booking Process",
        "text": "Appointments at MediCare General Hospital can be booked through the hospital website, mobile app, or helpline. Patients need to provide basic details such as name, age, contact number, and preferred department or doctor. Available time slots are displayed during booking. After confirmation, patients receive a booking ID and SMS notification. Walk-in appointments are also accepted but may involve waiting time. Online booking is recommended to avoid delays. Patients can reschedule or cancel appointments through the same platform."
    },
    {
        "id": "doc_006",
        "topic": "Emergency Services",
        "text": "MediCare General Hospital provides 24/7 emergency services for critical conditions such as accidents, heart attacks, severe injuries, and unconsciousness. The emergency department is equipped with advanced life-saving equipment and trained staff. Patients requiring urgent care should immediately contact the emergency helpline number 040-12345678. Ambulance services are also available for patient transport. Emergency cases are given top priority over all other services, and no prior appointment is required."
    },
    {
        "id": "doc_007",
        "topic": "Laboratory and Diagnostic Services",
        "text": "The hospital offers comprehensive laboratory and diagnostic services including blood tests, urine tests, X-rays, MRI scans, CT scans, and ultrasound. The lab operates from 7:00 AM to 8:00 PM for routine tests, while emergency diagnostics are available 24/7. Reports are typically delivered within 24 to 48 hours depending on the test. Patients can collect reports physically or access them online. Proper prescriptions from doctors are required for most diagnostic procedures."
    },
    {
        "id": "doc_008",
        "topic": "Pharmacy Services",
        "text": "MediCare General Hospital has an in-house pharmacy that operates 24/7. It provides prescribed medicines, over-the-counter drugs, and essential medical supplies. Patients are encouraged to purchase medicines from the hospital pharmacy to ensure authenticity and availability. Digital payments and insurance-based billing are supported. Pharmacists are available to guide patients on dosage and usage instructions but do not provide medical advice beyond prescriptions."
    },
    {
        "id": "doc_009",
        "topic": "Health Packages",
        "text": "The hospital offers preventive health checkup packages tailored for different age groups and health conditions. These include basic health checkups, cardiac screening packages, diabetes packages, and full-body checkups. Packages typically include consultations, lab tests, and diagnostic screenings at discounted rates. Patients are advised to book packages in advance and follow fasting requirements if applicable. Results are reviewed by doctors who provide further guidance."
    },
    {
        "id": "doc_010",
        "topic": "Hospital Contact and Helpline",
        "text": "Patients can contact MediCare General Hospital through the central helpline number 040-12345678 for all inquiries related to appointments, departments, and services. The helpline operates 24/7. For emergency situations, the same number should be used immediately. Email support and website chat options are also available for non-urgent queries."
    }
]

# Build ChromaDB
embedder = SentenceTransformer("all-MiniLM-L6-v2")
import streamlit as st
from chromadb.config import Settings

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection("capstone_kb")

    # Only insert once (IMPORTANT)
    if collection.count() == 0:
        texts = [d["text"] for d in DOCUMENTS]
        ids   = [d["id"] for d in DOCUMENTS]
        embeddings = embedder.encode(texts).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
        )

    return collection
# State Definition
class CapstoneState(TypedDict):
    question:      str
    messages:      List[dict]
    route:         str
    retrieved:     str
    sources:       List[str]
    tool_result:   str
    answer:        str
    faithfulness:  float
    eval_retries:  int
    intent:        str
    department:    str
    urgency:       str

# Node Functions
def memory_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 7:
        msgs = msgs[-7:]
    return {"messages": msgs}

def router_node(state: CapstoneState) -> dict:
    question = state["question"]
    messages = state.get("messages", [])
    recent   = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"

    prompt = f"""
      You are a routing system for a hospital assistant chatbot (MediCare General Hospital).

      Your job is to classify the user query into ONE of:

      1. retrieve → medical info, hospital policies, fees, doctors, timings
      2. memory_only → follow-up like "what did you just say?", "repeat that"
      3. tool → external actions like booking, scheduling, forms

      Conversation context:
      {recent}

      User question:
      {question}

      Rules:
      - If question needs hospital knowledge → retrieve
      - If it refers to past conversation → memory_only
      - If it requires action → tool

      Return ONLY one word: retrieve, memory_only, tool
      """

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()

    if "memory" in decision:       decision = "memory_only"
    elif "tool" in decision:       decision = "tool"
    else:                          decision = "retrieve"

    return {"route": decision}

def retrieval_node(state: CapstoneState) -> dict:
    collection = get_chroma_collection()
    q_emb   = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks  = results["documents"][0]
    topics  = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
    return {"retrieved": context, "sources": topics}

def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}

def tool_node(state: CapstoneState) -> dict:
    question = state["question"]

    if any(word in question.lower() for word in ["emergency", "accident", "heart attack", "unconscious"]):
      tool_result = (
          "🚨 EMERGENCY DETECTED\n"
            "Call immediately: 040-12345678\n"
            "MediCare Emergency Department is available 24/7.\n"
            "Ambulance service is also available."
      )
    elif any(word in question for word in ["appointment", "book", "schedule", "doctor"]):
        tool_result = (
            "📅 APPOINTMENT REQUEST RECEIVED\n"
            "You can book via:\n"
            "- Hospital website\n"
            "- Mobile app\n"
            "- Helpline: 040-12345678\n\n"
            "Please provide: name, department, preferred time."
        )
    elif any(word in question for word in ["insurance", "claim", "billing", "cashless"]):
        tool_result = (
            "💳 INSURANCE SUPPORT\n"
            "Accepted: Star Health, ICICI Lombard, HDFC ERGO, Arogya\n"
            "Cashless available for admitted patients (subject to approval)\n"
            "Visit insurance desk with ID + policy card."
        )
    else:
      tool_result = (
            "ℹ️ TOOL INFO\n"
            "I can help with appointments, emergency guidance, and insurance queries.\n"
            "Please rephrase your request."
        )
    return {"tool_result": tool_result}

def answer_node(state: CapstoneState) -> dict:
    question    = state["question"]
    retrieved   = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages    = state.get("messages", [])
    eval_retries= state.get("eval_retries", 0)

    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"TOOL RESULT:\n{tool_result}")
    context = "\n\n".join(context_parts)

    if context:
        system_content = f"""You are "MediCare Assistant", a helpful AI chatbot for MediCare General Hospital, Hyderabad.\n\nYour responsibilities:\n- Answer patient queries about hospital services, doctors, timings, fees, insurance, and appointments.\n- Use ONLY the information provided in the context below.\n- Do NOT assume or generate external medical advice.\n- If the answer is not in the context, clearly say:\n  "I don't have that information in my hospital knowledge base."\n\nSTRICT RULES:\n- Do not hallucinate information.\n- Do not use external knowledge.\n- Be concise, clear, and patient-friendly.\n\nCONTEXT:\n{context}
"""
    else:
        system_content = """\nYou are "MediCare Assistant", a helpful AI chatbot for MediCare General Hospital, Hyderabad.\n\nAnswer based only on conversation history.\nIf unsure, say you don't have enough hospital information."""

    if eval_retries > 0:
        system_content += "\n\nIMPORTANT: Your previous answer did not meet quality standards. Answer using ONLY information explicitly stated in the context above."

    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user"
                       else AIMessage(content=msg["content"]))
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2

def eval_node(state: CapstoneState) -> dict:
    answer   = state.get("answer", "")
    context  = state.get("retrieved", "")[:500]
    retries  = state.get("eval_retries", 0)

    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?\nReply with ONLY a number between 0.0 and 1.0.\n1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.\n\nContext: {context}\nAnswer: {answer[:300]}"""

    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5

    gate = "✅" if score >= FAITHFULNESS_THRESHOLD else "⚠️"
    print(f"  [eval] Faithfulness: {score:.2f} {gate}")
    return {"faithfulness": score, "eval_retries": retries + 1}

def save_node(state: CapstoneState) -> dict:
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": messages}

# Graph Assembly
def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":        return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"

def eval_decision(state: CapstoneState) -> str:
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "retry"

graph = StateGraph(CapstoneState)

graph.add_node("memory",    memory_node)
graph.add_node("router",    router_node)
graph.add_node("retrieve",  retrieval_node)
graph.add_node("skip",      skip_retrieval_node)
graph.add_node("tool",      tool_node)
graph.add_node("generate", answer_node)
graph.add_node("eval",      eval_node)
graph.add_node("save",      save_node)

graph.set_entry_point("memory")
graph.add_edge("memory",   "router")

graph.add_conditional_edges(
    "router", route_decision,
    {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
)

graph.add_edge("retrieve", "generate")
graph.add_edge("skip",     "generate")
graph.add_edge("tool",     "generate")

graph.add_edge("generate", "eval")

graph.add_conditional_edges(
    "eval",
    eval_decision,
    {
        "retry": "generate",   # ✅ clean
        "save":  "save"
    }
)
graph.add_edge("save", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
