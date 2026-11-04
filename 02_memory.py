"""第 2 課：上下文記憶。

模型本身不記得任何事，每次呼叫都是全新的。
LangGraph 的 checkpointer 會在每次呼叫時：
  1. 用 thread_id 找出這段對話的歷史訊息
  2. 把歷史 + 新訊息一起送給模型
  3. 把這次的問題和回答寫回歷史
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
SYSTEM = SystemMessage("你是友善的助理，用繁體中文、一句話回答。對話裡沒提過的事就說不知道，不要猜。")


print("=== A. 沒有記憶：兩次呼叫互不相干 ===")
for question in ["我叫 Rex，最喜歡的語言是 Swift", "我叫什麼？最喜歡什麼語言？"]:
    print("你：", question)
    print("AI：", llm.invoke([SYSTEM, HumanMessage(question)]).content)


# MessagesState 的 state 裡只有一個 messages 欄位，新訊息會自動「附加」上去，而不是覆蓋
def call_model(state: MessagesState) -> dict:
    print([SYSTEM] + state["messages"])  # 看看實際送給模型的內容
    print("=== 呼叫模型 ===")
    reply = llm.invoke([SYSTEM] + state["messages"])  # state["messages"] 就是完整歷史
    return {"messages": [reply]}


graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_edge(START, "model")

# checkpointer 負責保存每個 thread 的 state；InMemorySaver 存在記憶體，程式結束就消失
app = graph.compile(checkpointer=InMemorySaver())


def ask(thread_id: str, question: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(question)]}, config)  # 只送新問題，歷史它自己補
    print(f"[{thread_id}] 你：{question}")
    print(f"[{thread_id}] AI：{result['messages'][-1].content}")


print("\n=== B. 有記憶：同一個 thread 記得前面說過的話 ===")
ask("rex", "我叫 Rex，最喜歡的語言是 Swift")
ask("rex", "我叫什麼？最喜歡什麼語言？")

print("\n=== C. 換一個 thread：amy 的歷史是空的，所以不知道 ===")
ask("amy", "我叫什麼？最喜歡什麼語言？")

print("\n=== D. LangGraph 幫 rex 記下來的內容 ===")
for message in app.get_state({"configurable": {"thread_id": "rex"}}).values["messages"]:
    print(f"{message.type:>5}：{message.content}")
