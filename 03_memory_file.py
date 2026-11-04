"""第 3 課：把記憶存到檔案。

流程：
  第一次（沒有 user.conf）：輸入名字和性別 → 存成 user.conf
  每一次：從 memory.db 撈出這個人的身份和過去的問答 → 模型問一個新問題 → 你回答 → 存回 memory.db

跟第 2 課的差別：
  1. checkpointer 換成 SqliteSaver，state 存進 memory.db，重開程式還在
  2. state 除了 messages，還多記了 name、gender 兩個欄位
"""
import configparser
import os
import sqlite3

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph

CONF_PATH = "user.conf"
DB_PATH = "memory.db"

# temperature 調高，每次問的問題才會不一樣
llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.9)


class ProfileState(MessagesState):
    """在 messages 之外，多記使用者的身份。沒有 reducer，新值直接覆蓋舊值。"""
    name: str
    gender: str


def ask_question(state: ProfileState) -> dict:
    system = SystemMessage(
        f"你在認識一位使用者：名字是 {state['name']}，性別是 {state['gender']}。"
        "請用繁體中文問他一個簡短的生活問題，例如興趣、喜歡的食物、週末做什麼。"
        "不要重複之前問過的問題，只輸出問題本身。"
    )
    question = llm.invoke([system] + state["messages"])
    return {"messages": [question]}


graph = StateGraph(ProfileState)
graph.add_node("ask_question", ask_question)
graph.add_edge(START, "ask_question")
app = graph.compile(checkpointer=SqliteSaver(sqlite3.connect(DB_PATH, check_same_thread=False)))


# ---- 1. 讀取或建立 user.conf ----
conf = configparser.ConfigParser()
if os.path.exists(CONF_PATH):
    conf.read(CONF_PATH, encoding="utf-8")
    print(f"讀到 {CONF_PATH}，歡迎回來！")
else:
    print(f"找不到 {CONF_PATH}，先認識一下你。")
    conf["user"] = {"name": input("名字：").strip(), "gender": input("性別：").strip()}
    with open(CONF_PATH, "w", encoding="utf-8") as f:
        conf.write(f)
    print(f"已存到 {CONF_PATH}")

name, gender = conf["user"]["name"], conf["user"]["gender"]
config = {"configurable": {"thread_id": name}}  # 用名字當 thread_id，一個人一份記錄


# ---- 2. 從 memory.db 撈出過去的記錄 ----
saved = app.get_state(config).values
if saved:
    print(f"\n=== memory.db 裡關於 {saved['name']}（{saved['gender']}）的記錄 ===")
    messages = saved["messages"]
    for q, a in zip(messages[0::2], messages[1::2]):  # 訊息是「AI 問、你答」交替排列
        print(f"問：{q.content}")
        print(f"答：{a.content}\n")
else:
    print(f"\nmemory.db 裡還沒有 {name} 的記錄")


# ---- 3. 模型問一個新問題，你回答，存回 memory.db ----
result = app.invoke({"name": name, "gender": gender}, config)  # 身份寫進 state，順便觸發 ask_question
print("\nAI 問：", result["messages"][-1].content)
answer = input("你答：").strip()

# 回答不需要再問模型，直接用 update_state 把它附加到 state 裡
app.update_state(config, {"messages": [HumanMessage(answer)]})
print(f"\n已存到 {DB_PATH}，下次啟動會看到這筆記錄")
