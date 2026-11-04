# LangChain 入門練習

用本機 Ollama 的 `qwen2.5-coder:7b` 跑 LangChain。

## 安裝

1. 安裝 Ollama 並下載模型：`ollama pull qwen2.5-coder:7b`
2. 建立 conda 環境：`conda create -n langchaindemo python=3.12 -y`
3. 啟用環境：`conda activate langchaindemo`
4. 安裝套件：`pip install -r requirements.txt`

## 練習

- `01_hello.py`：呼叫模型（`ChatOllama`、`invoke`）
- `02_memory.py`：上下文記憶（LangGraph 的 `InMemorySaver`、`thread_id`）
- `03_memory_file.py`：把記憶存到 SQLite（`SqliteSaver`、自訂 state、`update_state`），身份存在 `user.conf`
- `04_loop_agent.py`：Loop engineering，收集資料直到齊全（`add_conditional_edges`、驗證、上限、錯誤回饋）

```bash
python 01_hello.py
```
