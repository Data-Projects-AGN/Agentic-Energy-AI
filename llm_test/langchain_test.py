import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory


load_dotenv()

def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    if "<think>" in text:
        return ""  # only thoughts were produced; next steps will fix via tokens
    return text

llm = ChatOpenAI(
    base_url=os.environ["JETSTREAM_BASE"],
    api_key=os.environ["JETSTREAM_API_KEY"],
    model=os.environ["JETSTREAM_MODEL"],
    temperature=0.2,
)

# messages = [
#     SystemMessage(content="Return only the final answer. No <think> text."),
#     HumanMessage(content="Hi")
# ]

# llm_response  = strip_think(llm.invoke(messages).content)
# print(llm_response)

# history = [SystemMessage(content="Return only the final answer. No <think> text.")]

# print("CLI ready. Ctrl+C to exit.")
# while True:
#     try:
#         u = input("> ").strip()
#         if not u: 
#             continue
#         history.append(HumanMessage(content=u))
#         out = llm.invoke(history).content
#         clean = strip_think(out)
#         print(clean)
#         history.append(AIMessage(content=clean))  # store sanitized text only
#     except KeyboardInterrupt:
#         print("\nBye")
#         break

prompt = ChatPromptTemplate.from_messages([
    ("system", "Return only the final answer. No <think> text."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# prompt -> llm -> to string -> sanitize; the output of this chain is what gets saved to history
chain = prompt | llm | StrOutputParser() | RunnableLambda(strip_think)

store: dict[str, ChatMessageHistory] = {}
def get_history(session_id: str) -> ChatMessageHistory:
    return store.setdefault(session_id, ChatMessageHistory())

chat = RunnableWithMessageHistory(
    chain,
    get_session_history=get_history,
    input_messages_key="input",      # the key you pass at invoke time
    history_messages_key="history",  # must match MessagesPlaceholder("history")
)

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