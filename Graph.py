# --- Imports ---
import json
from typing import TypedDict, Literal
import requests
from langchain.llms.base import LLM
from langgraph.graph import StateGraph, END

import SSearchF as searchVDB
import key

# --- Groq LLM Wrapper ---
class GroqLLM(LLM):
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    api_key: str = "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"  # Replace with your key
    base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.0

    def _call(self, prompt: str, stop: list[str] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "groq_custom_llm"


llm = GroqLLM()

# --- State Definition ---
class GraphState(TypedDict):
    input: str
    path: Literal["path1", "path2", "path3"]
    rag_context: str
    response: str


# --- Router Node ---
def router_node(state: GraphState) -> GraphState:
    prompt = f"""
You are a classifier. Based on the user input, pick exactly one of these paths:

"path1": user asks a general question (no code involved).
"path2": user asks a question and expects code in the answer.
"path3": Using The Provided Chuncks for referance ,Analyze the user code and identify any syntax errors, missing parts, or incorrect logic. Correct and complete the code so that it becomes functional and logically consistent. Maintain the intent of the original code (e.g., agent creation, tool integration, etc.). Return only the fixed code, followed by a brief explanation of the changes you made.

Respond ONLY with a JSON object exactly like this:

{{"path": "pathX"}}

User input:
{state['input']}
"""
    try:
        result_text = llm.invoke(prompt)
        parsed = json.loads(result_text.strip())
        state["path"] = parsed["path"]
    except Exception as e:
        print(f"Router failed: {e}")
        state["path"] = "path1"

    print(f"🧭 Path chosen: {state['path']}")
    return state


# --- Agent Nodes ---
def agent_path1_node(state: GraphState) -> GraphState:
    prompt = f"""You are a helpful assistant that answers general knowledge or conversational questions clearly and concisely.

Context:
{state['rag_context']}

User input:
{state['input']}"""

    response = llm.invoke(prompt)
    state["response"] = response
    return state


def agent_path2_node(state: GraphState) -> GraphState:
    prompt = f"""You are a code generation assistant. Always provide complete and executable Python code based on the user’s request. Include comments where helpful.

Context:
{state['rag_context']}

Task:
{state['input']}"""

    response = llm.invoke(prompt)
    state["response"] = response
    return state


def agent_path3_node(state: GraphState) -> GraphState:
    prompt = f"""You are a Python code expert. Explain or fix the provided code. Describe what it does, identify issues, and suggest corrections if needed based on the below context.

Context:
{state['rag_context']}

Code:
{state['input']}"""

    response = llm.invoke(prompt)
    state["response"] = response
    return state


# --- Conditional Router ---
def route_decision(state: GraphState) -> str:
    return state["path"]


# --- Build LangGraph ---
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("router", router_node)
workflow.add_node("path1", agent_path1_node)
workflow.add_node("path2", agent_path2_node)
workflow.add_node("path3", agent_path3_node)

# Route decision
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_decision, {
    "path1": "path1",
    "path2": "path2",
    "path3": "path3"
})

# End at all agent nodes
workflow.add_edge("path1", END)
workflow.add_edge("path2", END)
workflow.add_edge("path3", END)

# Compile the graph
app = workflow.compile()


# --- Entry Interface ---
def AgentSystem(user_input: str, ragout: str) -> str:
    state: GraphState = {
        "input": user_input,
        "path": "path1",  # Initial placeholder, updated by router
        "rag_context": ragout,
        "response": ""
    }

    result = app.invoke(state)
    print(f"\n💬 Final Response:\n{result['response']}")
    return result["response"]







# --- External Entry Point ---
def run_loop(ui_input: str):
    q = ui_input
    ragout = searchVDB.search_index(searchVDB.index, q)
    final_response = AgentSystem(q, ragout)
    return final_response, ragout


def run_loop(ui_input: str):
    q = ui_input

    # First: Run only the router to determine the path
    initial_state: GraphState = {
        "input": q,
        "path": "path1",  # default placeholder
        "rag_context": "",
        "response": ""
    }

    routed_state = router_node(initial_state.copy())
    chosen_path = routed_state["path"]

    if chosen_path == "path3":

        print("DID NORMAL SEARCH")
        ragout = searchVDB.search_index(searchVDB.index, q)

    else:
        ragout = searchVDB.search_index(searchVDB.index, key.get_keywords(q))
        print("DID ADVANCED SEARCH")


    print(f"\n🧭 Chosen path for input: '{q}' → {chosen_path}")


    # Prepare full state for execution
    full_state = {
        "input": q,
        "path": chosen_path,
        "rag_context": ragout,
        "response": ""
    }

    # Run the full agent graph
    result = app.invoke(full_state)

    #print(f"\n💬 Final Response:\n{result['response']}")
    return result["response"], ragout
