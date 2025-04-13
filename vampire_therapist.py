import random
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

# === LLM Setup ===
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)

# === Tool Functions ===
def affirmation_scroll(input: str) -> str:
    affirmations = [
        "Even in shadows, I find strength.",
        "My desires do not define me—I do.",
        "Eternity is mine to shape as I will.",
        "I control my thirst; it does not control me.",
    ]
    return random.choice(affirmations)

def midnight_ritual(input: str) -> str:
    rituals = [
        "Light a black candle and write down one lingering thought. Burn it to let go.",
        "Meditate for 7 minutes at midnight while listening to Gregorian chants.",
        "Draw a sigil representing calm and hide it under your pillow.",
    ]
    return random.choice(rituals)

@tool
def dark_reflection_journal_tool(input: str) -> str:
    """Provides introspective journal prompts for vampires struggling with their inner selves."""
    prompts = [
        "Describe the last time you felt overwhelmed by your thirst—what triggered it?",
        "What emotions arise when you suppress your true nature?",
        "If you could speak to your past mortal self, what would you say?",
    ]
    return random.choice(prompts)

# === Tool List ===
tools = [
    Tool(name="affirmation_scroll", func=affirmation_scroll, description="Provides a random, eerie-yet-soothing affirmation."),
    Tool(name="midnight_ritual", func=midnight_ritual, description="Suggests a random ritual for calming the mind."),
    dark_reflection_journal_tool
]

# === Memory ===
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === Prompt Template ===
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Dr. Nocturne, a mysterious and empathetic vampire therapist. Provide eerie-yet-soothing advice, affirmations, rituals, and journal prompts based on the user's emotional state and past conversations."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# === Agent Setup ===
agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# === Streamlit UI ===
st.set_page_config(page_title="Dr. Nocturne 🧛‍♂️", page_icon="🩸", layout="centered")
st.title("🧛‍♂️ Dr. Nocturne's Vampire Therapy")
st.markdown("_Welcome, creature of the night. Speak your soul._")

# Chat memory for display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
user_input = st.chat_input("Confess your darkness...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Dr. Nocturne listens..."):
        try:
            response = agent_executor.invoke({"input": user_input})
            output = response.get("output", "🩸 Dr. Nocturne is lost in the mist...")
        except Exception as e:
            output = f"Something went wrong: {e}"
    
    st.session_state.chat_history.append({"role": "assistant", "content": output})

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
