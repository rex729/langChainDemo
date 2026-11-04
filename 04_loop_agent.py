"""第 4 課：Loop engineering。

一個收集使用者資料的 loop agent，直到資料齊全才停：

    START → ask → extract → route ─資料齊全→ done
             ↑                    ├─還缺資料→ 回到 ask
             └────────────────────┘
                                  └─問太多次→ give_up

示範的 loop engineering：
  1. 停止條件由程式判斷（route），不是模型說了算
  2. 上限 MAX_TURNS，到了就收尾，不無限追問
  3. 每輪告訴模型「已知什麼、還缺什麼」
  4. 用程式驗證模型抽出的資料（年齡範圍、性別選項）
  5. 不合格時把原因回饋給下一輪
  6. 模型輸出壞掉時不當掉，當作這輪沒拿到資料
  7. 只送需要的資訊給模型，不送整段對話，上下文保持很短
  8. 兩個模型分工：asker 負責問、extractor 負責抽資料
"""
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

MAX_TURNS = 6
FIELDS = {"name": "名字", "gender": "性別", "age": "年齡", "job": "職業"}
GENDERS = {"男": "男", "男生": "男", "male": "男", "女": "女", "女生": "女", "female": "女", "其他": "其他"}

# 8. 兩個模型分工：問問題要自然一點，抽資料要穩定
asker = ChatOllama(model="qwen2.5-coder:7b", temperature=0.7)
extractor = ChatOllama(model="qwen2.5-coder:7b", temperature=0)


class Extracted(BaseModel):
    """從一句回答裡抽出的資料，沒提到的欄位是 None。"""
    name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    job: Optional[str] = None


class State(TypedDict):
    profile: dict       # 已經驗證通過的資料
    errors: list[str]   # 上一輪不合格的原因，回饋給下一輪
    question: str
    answer: str
    turn: int


def missing(profile: dict) -> list[str]:
    return [label for key, label in FIELDS.items() if key not in profile]


def ask(state: State) -> dict:
    known = "、".join(f"{FIELDS[k]}={v}" for k, v in state["profile"].items()) or "還沒有"
    # 3. 告訴模型目前的狀態；5. 帶上上一輪的錯誤原因
    # 7. 只給這些摘要，不給整段對話
    instruction = (
        f"你在幫使用者建立個人資料。已知：{known}。還缺：{'、'.join(missing(state['profile']))}。"
        + (f"上一輪的問題：{'；'.join(state['errors'])}。請針對這點再問一次。" if state["errors"] else "")
        + "用繁體中文問一個簡短、友善的問題，一次問一到兩個缺少的欄位，只輸出問題本身。"
    )
    question = asker.invoke([SystemMessage(instruction), HumanMessage("請問下一個問題")]).content.strip()
    turn = state["turn"] + 1
    print(f"\n[第 {turn} 輪] AI：{question}")
    return {"question": question, "answer": input("你：").strip(), "turn": turn}


def extract(state: State) -> dict:
    prompt = (
        f"問題：{state['question']}\n回答：{state['answer']}\n"
        "只抽出回答裡明確提到的資料，沒提到的欄位填 null，不要猜。"
    )
    try:
        data = extractor.with_structured_output(Extracted).invoke(prompt)
    except Exception as e:  # 6. 模型輸出壞掉，不讓整個 loop 當掉
        print(f"  [harness] 抽資料失敗（{type(e).__name__}），這輪當作沒拿到")
        return {"errors": ["沒聽懂剛剛的回答"]}

    profile, errors = dict(state["profile"]), []
    # 4. 模型抽出來的值，由程式再檢查一次
    if data.name and data.name.strip():
        profile["name"] = data.name.strip()
    if data.gender:
        if data.gender.strip().lower() in GENDERS:
            profile["gender"] = GENDERS[data.gender.strip().lower()]
        else:
            errors.append(f"性別「{data.gender}」看不懂，請回答男、女或其他")
    if data.age is not None:
        if 1 <= data.age <= 120:
            profile["age"] = data.age
        else:
            errors.append(f"年齡 {data.age} 不合理")
    if data.job and data.job.strip():
        profile["job"] = data.job.strip()

    print(f"  [harness] 抽到：{data.model_dump(exclude_none=True)}")
    print(f"  [harness] 驗證通過：{profile}；還缺：{missing(profile) or '無'}"
          + (f"；不合格：{errors}" if errors else ""))
    return {"profile": profile, "errors": errors}


def route(state: State) -> str:
    # 1. 停止條件由程式判斷
    if not missing(state["profile"]):
        return "done"
    # 2. 上限
    if state["turn"] >= MAX_TURNS:
        return "give_up"
    return "ask"


def done(state: State) -> dict:
    print(f"\n✅ 資料齊全，共問了 {state['turn']} 輪：{state['profile']}")
    return {}


def give_up(state: State) -> dict:
    # 2. 優雅收尾：保留已經拿到的，清楚說明缺什麼
    print(f"\n⚠️ 已問 {MAX_TURNS} 輪，先存下目前的資料：{state['profile']}，還缺：{missing(state['profile'])}")
    return {}


graph = StateGraph(State)
graph.add_node("ask", ask)
graph.add_node("extract", extract)
graph.add_node("done", done)
graph.add_node("give_up", give_up)
graph.add_edge(START, "ask")
graph.add_edge("ask", "extract")
graph.add_conditional_edges("extract", route, ["ask", "done", "give_up"])  # loop 就在這條邊
graph.add_edge("done", END)
graph.add_edge("give_up", END)
app = graph.compile()

app.invoke({"profile": {}, "errors": [], "question": "", "answer": "", "turn": 0})
