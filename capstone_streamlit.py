import streamlit as st
import os
from agent import app, CapstoneState  # Import your compiled agent and State
from langgraph.checkpoint.memory import MemorySaver

# Assuming `app` is compiled with a checkpointer like MemorySaver
# If you're running this as a standalone script, you might need to re-initialize it.
# For Streamlit, we want the checkpointer to persist across reruns.

# Initialize MemorySaver outside of the main run to persist across sessions
# Streamlit's session state can help manage this.
if 'checkpointer' not in st.session_state:
    st.session_state.checkpointer = MemorySaver()

# Initialize the agent with the session-specific checkpointer
# This ensures each user session has its own memory
def get_agent():
    return app  # The compiled app from agent.py already has the checkpointer

agent = get_agent()

st.title("MediCare Assistant Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about MediCare Hospital services..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the agent
    # Use a unique thread_id for each Streamlit session if you want separate conversations
    # For simplicity, we'll use a fixed 'streamlit_thread' here.
    # In a real multi-user deployment, you'd generate a unique ID per user.
    thread_id = "streamlit_thread"
    config = {"configurable": {"thread_id": thread_id}}

    # The question is the prompt from the user
    agent_input = {"question": prompt, "messages": st.session_state.messages[:-1]} # Pass all but the current user message for context if needed by nodes

    # The first turn needs to just pass the question. Subsequent turns will use full history.
    # The `memory_node` in agent.py will handle adding the current question to the history.

    # If the app is designed to handle history internally, we just pass the question.
    # The `memory_node` handles adding the question to the messages list.
    # If memory_node expects messages from the *start* of the turn, the agent input should be clean.
    # Let's adjust the agent input to be just the current question.

    # When interacting with a StateGraph, the input usually reflects the `CapstoneState` schema.
    # The `ask` helper in the notebook might have abstracted this.
    # Here, `messages` in `CapstoneState` is *conversation history before this turn* + *current user question*
    # The memory_node is designed to *add* the current question to existing messages.
    # So, the input to the agent should only contain `question` and current `messages` without the last one.

    result = agent.invoke(agent_input, config=config)
    agent_response = result.get("answer", "Sorry, I couldn't process that.")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": agent_response})
    with st.chat_message("assistant"):
        st.markdown(agent_response)

# Optional: Display current conversation history state for debugging
# st.sidebar.expander("Conversation History").json(st.session_state.messages)
