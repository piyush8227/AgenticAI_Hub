import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    code: str
    final_code: str
    improvements: str
    doc_string: str

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile")
    
# Workflow
def agent_call():
    """Structure"""

    def generate_code(state: State):
        """First LLM call to generate initial code"""

        msg = llm.invoke(f"Write a pyhton code using Object Oriented Programing concepts for this problem state: {state['query']}. If after the peer review, if you got any {state['improvements']} then please make neccesary updates in code.")
        return {"code": msg.content}

    def check_result(state: State):
        """Second LLM call for this Gate function to check whether reviewer has Passed the code or declared it as Fail."""
        msg = llm.invoke(f"Act as a Peer Reviewer and please make a review on this: {state['code']}, check the code efficieny and is it able to solve this problem statement: {state['query']} and return boolean value, either 'Pass' or 'Fail' after review.")

        if "Fail" in msg.content:
            return "Fail"
        return {"Pass"}

    def peer_review(state:State):
        """3rd LLM Call for making code fixes that reviewer has asked to do."""
        msg = llm.invoke(f"As the peer reviewer, please make a code review, check the code efficieny and is it able to solve this problem statement, and give points that are missing or are needed to be added in the {state['code']}.")
        return {"improvements": msg.content}


    def manager_approval(state: State):
        """5th LLM call for manager to prepare a doc_string for the final code."""

        msg = llm.invoke(f"Act as a project manager and make a well documented doc string for the {state['code']}.")
        return {"doc_string": msg.content}
    
    workflow = StateGraph(State)

    # Nodes
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("peer_review", peer_review)
    workflow.add_node("manager_approval", manager_approval)

    # Edges
    workflow.add_edge(START, "generate_code")
    workflow.add_edge("generate_code","peer_review")
    workflow.add_conditional_edges("peer_review", check_result, {"Fail":"generate_code", "Pass": "manager_approval"})
    workflow.add_edge("peer_review", "manager_approval")
    workflow.add_edge("manager_approval", END)

    # Compile
    graph = workflow.compile()
    return graph

graph = agent_call()