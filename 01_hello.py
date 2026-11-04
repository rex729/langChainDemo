"""第 1 課：直接呼叫模型。

ChatOllama 把本機的 Ollama 包成 LangChain 的「聊天模型」，
之後換成 OpenAI、Claude 也只要換這一行，其餘程式碼不用動。
"""
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

# invoke() 丟一句話進去，拿回一個 AIMessage
reply = llm.invoke("用一句話解釋什麼是 LangChain，用繁體中文回答")
print(type(reply).__name__)  # AIMessage
print(reply.content)
