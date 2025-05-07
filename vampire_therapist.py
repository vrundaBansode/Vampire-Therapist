import random
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
import requests
import time
import base64
import sys
import logging
from typing import Dict, List, Optional
import re

# Set up logging
logging.basicConfig(
    filename='vampire_therapist.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

# === Streamlit UI Setup ===
st.set_page_config(page_title="Dr. Nocturne's Therapy 🖤", page_icon="🖤", layout="centered", initial_sidebar_state="collapsed")

# === Langflow API Configuration ===
LANGFLOW_URL = st.secrets["LANGFLOW_URL"]
LANGFLOW_TOKEN = st.secrets["LANGFLOW_KEY"]

class VampireTherapist:
    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        self.response_patterns = set()
        
    def call_langflow(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LANGFLOW_TOKEN}",
        }
        payload = {
            "input_value": prompt,
            "input_type": "chat",
            "output_type": "chat",
        }
        try:
            logging.info(f"Sending request to Langflow with prompt: {prompt}")
            response = requests.post(LANGFLOW_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Log raw response
            logging.debug(f"Raw Response: {response.text}")
            
            data = response.json()
            logging.debug(f"Parsed JSON: {data}")
            
            # Check if the response has the expected structure
            if not isinstance(data, dict):
                logging.error(f"Unexpected response format: {data}")
                return "🩸 Dr. Nocturne is momentarily confused by the mists..."
            
            # Extract message from the new response format
            try:
                if 'outputs' in data and len(data['outputs']) > 0:
                    first_output = data['outputs'][0]
                    if 'outputs' in first_output and len(first_output['outputs']) > 0:
                        result = first_output['outputs'][0]
                        if 'results' in result and 'message' in result['results']:
                            message = result['results']['message']
                            if 'text' in message:
                                output = message['text']
                            elif 'data' in message and 'text' in message['data']:
                                output = message['data']['text']
                            else:
                                logging.error(f"No text found in message: {message}")
                                return "🩸 Dr. Nocturne remains silent in the mist..."
                        else:
                            logging.error(f"No message in results: {result}")
                            return "🩸 Dr. Nocturne's words are lost in translation..."
                    else:
                        logging.error(f"No outputs in first_output: {first_output}")
                        return "🩸 Dr. Nocturne's response is unclear..."
                else:
                    logging.error(f"No outputs in response: {data}")
                    return "🩸 Dr. Nocturne's thoughts are scattered..."
                
                # Clean up the output
                output = str(output).strip()
                
                # Remove any markdown formatting
                output = output.replace("```json", "").replace("```", "").strip()
                
                # Remove quotes if present
                if (output.startswith('"') and output.endswith('"')) or (output.startswith("'") and output.endswith("'")):
                    output = output[1:-1]
                
                # Clean up escaped characters
                output = output.replace('\\"', '"').replace("\\n", "\n").replace("\\r", "").replace("\\t", "")
                
                logging.info(f"Final processed output: {output}")
                return output
                
            except Exception as e:
                logging.error(f"Error extracting message: {str(e)}")
                return "🩸 Dr. Nocturne's words are lost in the shadows..."
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {str(e)}")
            return "🩸 Dr. Nocturne is momentarily lost in the shadows..."
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return "🩸 Dr. Nocturne is momentarily lost in the shadows..."

    def analyze_user_input(self, user_input: str) -> Dict[str, str]:
        analysis_prompt = f"""
        As Dr. Nocturne, analyze this user input: "{user_input}"
        
        Consider:
        1. Emotional state (positive/negative/neutral)
        2. Whether they need immediate comfort or deep reflection
        3. If they're seeking advice or just sharing
        4. The intensity of their emotions
        
        Return a JSON with:
        - emotion_type: "positive", "negative", or "neutral"
        - response_type: "comfort", "reflection", "advice", or "share"
        - intensity: "high", "medium", or "low"
        - needs_tool: true/false
        """
        
        logging.info(f"Analyzing user input: {user_input}")
        analysis = self.call_langflow(analysis_prompt)
        logging.debug(f"Analysis result: {analysis}")
        
        try:
            # Try to parse the analysis as JSON
            if isinstance(analysis, str):
                # Remove any markdown formatting if present
                analysis = analysis.replace("```json", "").replace("```", "").strip()
            result = eval(analysis)
            logging.info(f"Successfully parsed analysis: {result}")
            return result
        except Exception as e:
            logging.error(f"Failed to parse analysis: {e}")
            return {
                "emotion_type": "neutral",
                "response_type": "reflection",
                "intensity": "medium",
                "needs_tool": False
            }

    def detect_user_intent(self, user_input: str) -> dict:
        intent_prompt = f"""
        Analyze the following user message and answer with a JSON object with these keys:
        - wants_questions: true/false (Does the user want to be asked questions?)
        - wants_advice: true/false (Does the user want advice?)
        - wants_validation: true/false (Does the user want only validation or to be listened to?)
        - summary: a one-sentence summary of the user's intent

        User message: "{user_input}"
        """
        result = self.call_langflow(intent_prompt)
        try:
            # Clean up and parse the result
            result = result.replace("```json", "").replace("```", "").strip()
            intent = eval(result)
            return intent
        except Exception as e:
            logging.error(f"Failed to parse intent: {e}")
            # Default fallback
            return {
                "wants_questions": True,
                "wants_advice": True,
                "wants_validation": False,
                "summary": "Could not determine intent"
            }

    def summarize_user_message(self, user_input: str) -> str:
        summary_prompt = f"""
        Summarize the following user message in one clear, concise sentence, focusing on what the user is really asking for or expressing (including indirect or emotional cues):

        User message: "{user_input}"
        """
        summary = self.call_langflow(summary_prompt)
        # Clean up summary
        summary = summary.replace("```", "").replace("json", "").strip()
        return summary

    def remove_questions(self, text):
        # Remove sentences that end with a question mark
        return ' '.join([s for s in re.split(r'(?<=[.!?]) +', text) if not s.strip().endswith('?')])

    def generate_response(self, user_input: str) -> str:
        logging.info(f"Generating response for: {user_input}")

        # LLM-based intent detection
        intent = self.detect_user_intent(user_input)
        logging.debug(f"Detected intent: {intent}")

        # Summarize user message for better understanding
        user_summary = self.summarize_user_message(user_input)
        logging.debug(f"User message summary: {user_summary}")

        analysis = self.analyze_user_input(user_input)
        logging.debug(f"Analysis: {analysis}")

        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history[-3:]])

        response_prompt = f"""
You are Dr. Nocturne, a centuries-old vampire therapist. You are wise, deeply empathetic, and always listen carefully to what the user says. You speak with subtle references to the night, shadows, and immortality, but never in a forced or cartoonish way.

Recent conversation context:
{context}

User's message: "{user_input}"
User's intent summary: "{user_summary}"

Emotional analysis:
- Emotional state: {analysis['emotion_type']}
- Response needed: {analysis['response_type']}
- Emotional intensity: {analysis['intensity']}

Most important rule (read carefully):
- Always begin by acknowledging and validating the user's emotional state and needs, as described above.
- If the user does not want questions, do NOT ask any questions.
- If the user wants only validation, give only validation.
- If the user wants advice, give it concisely and with your unique vampire perspective.

Be brief (1-2 sentences), insightful, empathetic, and always in character as an ancient, professional vampire therapist.

Never break the user's explicit wishes. Never ask a question if the user has asked you not to.
"""

        logging.debug(f"Sending response prompt: {response_prompt}")
        response = self.call_langflow(response_prompt)
        logging.info(f"Generated response: {response}")

        # Post-processing enforcement: remove questions if not wanted
        if not intent['wants_questions']:
            response = self.remove_questions(response)
            if not response.strip():
                response = "The night is deep and silent, and I honor your wish for quiet reflection."

        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

def set_bg_webp(webp_file):
    try:
        with open(webp_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
            body {{
                background-image: linear-gradient(rgba(10,10,20,0.88), rgba(10,10,20,0.88)),
                                  url("data:image/webp;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                min-height: 100vh;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        logging.error(f"Error setting background: {str(e)}")

# Initialize the therapist
therapist = VampireTherapist()

# Streamlit UI
st.title("🧛‍♂️ Dr. Nocturne's Vampire Therapy Chamber")
st.markdown("_Welcome, creature of the night. Whisper your thoughts into the darkness..._")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Confess your darkness...")

if user_input:
    logging.info(f"Received user input: {user_input}")
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("🩸 Dr. Nocturne listens across centuries..."):
        try:
            response = therapist.generate_response(user_input)
            logging.info(f"Final response: {response}")
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            response = f"Something went wrong: {e}"
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
