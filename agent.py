"""Lesson 2: the Arabic reading-assistant agent.

LangChain's create_agent (built on LangGraph) runs the classic agent loop:
  1. the LLM (OpenAI) reads the conversation
  2. it decides to call a tool -> search_books(query)
  3. the tool result goes back to the LLM
  4. the LLM writes the final answer in Arabic

Usage (in CloudShell):
  read -s OPENAI_API_KEY && export OPENAI_API_KEY   # typed hidden, never committed
  export OPENAI_MODEL=<a model id your key can use>
  python agent.py
"""
import json
import os

from langchain.agents import create_agent
from langchain.tools import tool

from prompts import PROMPTS
from recommender import Recommender

MODEL_ID = os.environ.get("OPENAI_MODEL", "")

_recommender = Recommender()


@tool
def search_books(query: str) -> str:
    """Search the book catalog (200 Arabic books) for books matching a reader's request.

    Pass a short Arabic query describing the topic, genre or mood, for example
    "رواية تاريخية" or "كتب عن الإدارة". Returns a JSON list of up to 5 books with
    id, title, author, category, a short description and a similarity score.
    Only books in this list exist in the catalog.
    """
    return json.dumps(_recommender.search(query, k=5), ensure_ascii=False)


def make_model():
    """The LLM brain: an OpenAI chat model. The key is read from OPENAI_API_KEY."""
    from langchain_openai import ChatOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first:  read -s OPENAI_API_KEY && export OPENAI_API_KEY")
    if not MODEL_ID:
        raise SystemExit("Set OPENAI_MODEL first (Lesson 2: list the models your key can use).")
    return ChatOpenAI(model=MODEL_ID)  # model defaults: some new models reject a custom temperature


def build_agent(prompt_version: str = "v1", model=None):
    return create_agent(
        model or make_model(),
        tools=[search_books],
        system_prompt=PROMPTS[prompt_version],
    )


def text_of(message) -> str:
    """Content may be a plain string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def langfuse_callbacks() -> list:
    """Trace to Langfuse when its keys are set; otherwise run without tracing."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def ask(agent, question: str, callbacks: list | None = None) -> dict:
    """Run one question through the agent.

    Returns the answer, the tool calls it made, and the titles the tool actually
    retrieved (the evals need these to judge the answer fairly).
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": callbacks or []},
    )
    tool_calls, retrieved = [], []
    for m in result["messages"]:
        tool_calls += [call["args"] for call in (getattr(m, "tool_calls", None) or [])]
        if getattr(m, "type", "") == "tool":
            try:
                retrieved += [b["title"] for b in json.loads(text_of(m))]
            except (ValueError, TypeError, KeyError):
                pass
    return {
        "answer": text_of(result["messages"][-1]),
        "tool_calls": tool_calls,
        "retrieved": list(dict.fromkeys(retrieved)),  # unique, keep order
    }


if __name__ == "__main__":
    agent = build_agent(os.environ.get("PROMPT_VERSION", "v1"))
    callbacks = langfuse_callbacks()
    print("Langfuse tracing:", "ON" if callbacks else "off (no LANGFUSE_PUBLIC_KEY)")
    print("مساعد القراءة — اكتب سؤالك (أو 'خروج' للإنهاء)\n")
    while True:
        question = input("أنت: ").strip()
        if question in {"", "خروج", "exit", "quit"}:
            break
        reply = ask(agent, question, callbacks=callbacks)
        for call in reply["tool_calls"]:
            print(f"  🔎 search_books({call})")
        print(f"\nالمساعد: {reply['answer']}\n")
