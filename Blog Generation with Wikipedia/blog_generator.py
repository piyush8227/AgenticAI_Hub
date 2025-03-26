# Libraries

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from IPython.display import Image, display, Markdown
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Literal
import os

# Api Key Call
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Defining State
class State(TypedDict):
    topic: str
    text: str
    blog: str
    feedback: str
    good_or_bad: str
    final_blog: str

# Defining Evaluator Criteria and Evaluator LLM 
class Feedback(BaseModel):

    # Evaluates whether the blog is Good or Bad
    eval: Literal["Good", "Bad"] = Field(
        description="Decide if the Blog is Good or Bad"
    )

    # Provides a detailed feedback upon considering factors.
    feedback: str = Field(
        description="""
        If the Blog is Bad, provide a feedback on how to improve it considering below mentioned factors to evalute blog.
        1. Content Quality: 
            a. Clarity & Readability: Well-structured, Easy to read & is in Active voice.
            b. Depth & Value: Gives In-depth analysis, has Actionable insights & Unique perspective.
            c. Engagement & Flow: Has Engaging storytelling, Smooth transitions & has Human Conversational tone.
            
        2. Accuracy & Relevance:
            a. Fact-Checking & Credibility: It should have Verified data, Proper citations, Expert-backed content.
            b. Industry Relevance: It is Trend-aligned, and Relevant to audience.
            
        3. SEO Optimization:
            a. Keyword Usage: Has Natural integration, and High-value keywords.
            b. Metadata & Structure: Well-optimized title & meta description, Proper header tags (H1, H2, H3).
            c. Internal & External Linking: Proper linking strategy, Credible sources.
        
        4. Engagement & User Experience:
            a. Visual Appeal: Is in Proper formatting (bullet points, lists).
            b. Call to Action (CTA): Clear CTA (subscribe, comment, share), it Encourages discussion.
            c. Readability Score (Flesch-Kincaid, Hemingway app, etc.): Easy-to-read.

        5. Performance Metrics (Post-Publishing Evaluation):
            a. User Engagement: High time spent on page, High comments/shares.
            b. Search Performance: Ranking on search engines, High organic traffic.
            c. Business Impact: Increased conversions/leads, Positive brand sentiment.
        """
    )

evaluator = llm.with_structured_output(Feedback)

# Information Extractor from Wikipedia

def information(state: State):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    data = wikipedia.run(state["topic"])
    msg = llm.invoke(f"""You are an AI agent designed to clean raw Wikipedia text by removing unnecessary elements while preserving meaningful content. 
                     Your goal is to extract clean, structured, and readable text suitable for analysis from {data}.
                    Cleaning Steps:

                    Remove Wikipedia Markups & Formatting:

                    Strip Wikitext elements like [[, ]], {{, }}, == Heading ==, etc.
                    Remove inline citations [1], <ref>...</ref>, and {{cite ...}}.
                    Remove URLs.
                    Eliminate Non-Textual Elements:

                    Remove Infoboxes ({{Infobox ... }}), tables, and image captions ([[File:...]]).
                    Strip Metadata & Categories:

                    Remove lines like [[Category:Science]], [[fr:Exemple]], and {{stub}}.
                    Exclude page edit notices like {{disambiguation}}.
                    Remove Unwanted Sections:

                    Identify and delete sections titled "See Also," "References," and "External Links".
                    Sanitize Special Characters & Formatting:

                    Remove excessive newlines, extra spaces, and HTML tags (<div>, <script>, etc.).
                    Convert Unicode artifacts into readable text.
                    Output Format:
                    Return clean, plain text without unnecessary symbols.
                    Preserve paragraph structures where possible.
            """)
    return {"text": msg.content}

# Blog Writter LLM 

def blog_writer(state: State):
    """Expert Blog Writer who writes engaging and informative blogs across various social media platforms."""
    if state.get("feedback"):
        msg = llm.invoke(f"Write a blog of 1000 words from this information provided: {state['text']} but take into account the feedback: {state['feedback']}")

    else:
        messages = [
            SystemMessage(content="You are an expert blog writer. Write engaging and informative blogs."),
            HumanMessage(content=f"Write a blog of 1000 words from this information provided: {state['text']}")
        ]
        msg = llm.invoke(messages)
    
    return {"blog": msg.content}

# Evaluator LLM 

def evaluator_llm(state: State):
    """LLM evaluates the blog."""
    grade = evaluator.invoke(f"Grade the blog {state['blog']}")
    return {"good_or_bad": grade.eval, "feedback": grade.feedback}

# Conditional Edge function to route back to Blog generator or End based upon feedback from evaluator

def route_blog(state: State):
    """Route back to Blog Writer or end based upon the feedback from the evaluator."""

    if state["good_or_bad"] == "Good":
        return "Accepted"
    elif state["good_or_bad"] == "Bad":
        return "Rejected + Feedback"
    
# Markdown formatter for providing blog in a good format. 

def markdown_formatter(state:State):
    return {"final_blog": Markdown(state["blog"])}

# Defining StateGraph
blog_builder = StateGraph(State)

# Graph Nodes

blog_builder.add_node("information", information)
blog_builder.add_node("blog_writer", blog_writer)
blog_builder.add_node("evaluator_llm", evaluator_llm)
blog_builder.add_node("markdown_formatter", markdown_formatter)

# Graph Edges

blog_builder.add_edge(START, "information")
blog_builder.add_edge("information", "blog_writer")
blog_builder.add_edge("blog_writer", "evaluator_llm")
blog_builder.add_conditional_edges(
    "evaluator_llm",
    route_blog,
    {
        "Accepted": "markdown_formatter",
        "Rejected + Feedback": "blog_writer"
    },
)
blog_builder.add_edge("markdown_formatter", END)

# Compiling our graph for workflow
graph = blog_builder.compile()