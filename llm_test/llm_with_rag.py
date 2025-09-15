import os, json
from dotenv import load_dotenv
from rag_get_pdf_data import search_pdfs

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

BASE = os.environ["JETSTREAM_BASE"]
KEY = os.environ["JETSTREAM_API_KEY"]
MODEL = os.environ["JETSTREAM_MODEL"]

def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    if "<think>" in text:
        return ""  # only thoughts were produced; next steps will fix via tokens
    return text
    

llm = ChatOpenAI(
    base_url=BASE,
    api_key=KEY,
    model=MODEL,
    temperature=0.2,
)


# @tool
# def search_pdfs(query: str, k: int = 5) -> str:
#     """Search the PDF vector store with the user's full question and return up to k short snippets,
#     each tagged with a [source_id]. Use when external facts from PDFs are required."""
#     # TODO: embed `query` with nomic-ai/nomic-embed-text-v1.5
#     # TODO: cosine search top-k over 768-dim vectors
#     # TODO: build compact text with [source_id] tags for each snippet
#     demo = (
#         "[DOC42:p12] Refunds must be requested within 30 days.\n"
#         "[DOC12:p4] Non-refundable items are listed in Section 3.\n"
#         "[DOC42:p13] Processing typically takes 5-7 business days."
#     )
#     return demo

# tool_llm = llm.bind_tools([search_pdfs])   # tool_choice="auto" by default
tool_llm = llm.bind_tools([search_pdfs])

prompt = ChatPromptTemplate.from_messages([
    ("system",
    "Answer from conversation when possible. "
    "If external facts from PDFs are needed, CALL the tool `search_pdfs` with the user's full question. "
    "After tools return, cite the [source_id]s and return only the final answer. No <think>. "
    "If context is insufficient, say you don't know."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])


TOOL_REGISTRY = {search_pdfs.name: search_pdfs}

def run_one_turn(inputs: dict) -> str:
    msgs = prompt.format_messages(input=inputs["input"], history=inputs.get("history", []))
    ai: AIMessage = tool_llm.invoke(msgs)

    if getattr(ai, "tool_calls", None):
        tool_msgs = []
        for tc in ai.tool_calls:
            args = tc.get("args", {}) if isinstance(tc, dict) else tc.args
            # run the tool (returns JSON string with joined_context + passages)
            result = search_pdfs.invoke(args or {"query": inputs["input"]})
            tool_msgs.append(ToolMessage(content=result, tool_call_id=tc["id"] if isinstance(tc, dict) else tc.id))
        # include the assistant tool-call message AND tool outputs
        msgs = msgs + [ai] + tool_msgs
        ai = tool_llm.invoke(msgs)

    return strip_think(ai.content)

# def run_one_turn(inputs: dict) -> str:
#     # inputs carries {"input": user_text, "history": [...past BaseMessage objects...] }
#     msgs = prompt.format_messages(input=inputs["input"], history=inputs.get("history", []))

#     ai: AIMessage = tool_llm.invoke(msgs)

#     loops = 0
#     max_loops = 1   # keep it to a single tool round-trip per turn for now
#     while getattr(ai, "tool_calls", None) and loops < max_loops:
#         tool_msgs = []
#         for tc in ai.tool_calls:
#             name = tc["name"] if isinstance(tc, dict) else tc.name
#             args = tc["args"] if isinstance(tc, dict) else tc.args
#             call_id = tc["id"] if isinstance(tc, dict) else tc.id

#             tool_obj = TOOL_REGISTRY.get(name)
#             if tool_obj is None:
#                 result_text = f"[ERROR] Unknown tool: {name}"
#             else:
#                 # In real code, args is a dict like {"query": "...", "k": 5}
#                 result_text = tool_obj.invoke(args)

#             tool_msgs.append(ToolMessage(content=result_text, tool_call_id=call_id))

#         # msgs.extend(tool_msgs)
#         msgs = msgs + [ai] + tool_msgs
#         ai = tool_llm.invoke(msgs)
#         loops += 1

#     text = ai.content if isinstance(ai, AIMessage) else str(ai)
#     return strip_think(text)


store: dict[str, ChatMessageHistory] = {}
def get_history(session_id: str) -> ChatMessageHistory:
    return store.setdefault(session_id, ChatMessageHistory())

controller = RunnableLambda(run_one_turn)

chat = RunnableWithMessageHistory(
    controller,
    get_session_history=get_history,
    input_messages_key="input",       # must match {input} in the prompt
    history_messages_key="history",   # must match MessagesPlaceholder("history")
)

if __name__ == "__main__":
    session_id = "cli"
    print("CLI ready. Ctrl+C or Ctrl+D to exit.")
    while True:
        try:
            user = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye")
            break
        if not user:
            continue
        result = chat.invoke(
            {"input": user},
            config={"configurable": {"session_id": session_id}}
        )
        print(result)
