"""Lesson 4: the reading assistant as a small web app (FastAPI).

  GET  /        Arabic chat page (RTL)
  POST /chat    {"question": "..."} -> {"answer": "...", "searches": [...]}
  GET  /health  "ok"  (the load balancer checks this)

Run locally:  uvicorn app:app --port 8080
Env vars (in AWS they come from Secrets Manager, never from code):
  OPENAI_API_KEY, OPENAI_MODEL, PROMPT_VERSION (default v2),
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL (optional tracing)
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

import agent

PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v2")
the_agent = agent.build_agent(PROMPT_VERSION)  # built once, reused by every request

app = FastAPI(title="Reading Assistant")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.post("/chat")
def chat(q: Question) -> dict:
    # A plain def runs in FastAPI's thread pool, so one slow answer doesn't block others.
    try:
        reply = agent.ask(the_agent, q.question.strip(), callbacks=agent.langfuse_callbacks())
    except Exception as exc:  # show a friendly error, keep details in the logs
        print("chat error:", repr(exc), flush=True)
        raise HTTPException(status_code=502, detail="تعذّر الحصول على إجابة الآن، حاول مرة أخرى.")
    return {"answer": reply["answer"], "searches": [c.get("query", "") for c in reply["tool_calls"]]}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


PAGE = """<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>مساعد القراءة</title>
<style>
  body { font-family: system-ui, "Segoe UI", Tahoma, sans-serif; background: #f6f4ef; margin: 0; color: #222; }
  main { max-width: 720px; margin: 0 auto; padding: 24px 16px; }
  h1 { margin: 0 0 4px; font-size: 1.6rem; }
  p.sub { margin: 0 0 20px; color: #666; }
  #log { display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; }
  .msg { padding: 12px 14px; border-radius: 12px; white-space: pre-wrap; line-height: 1.7; }
  .me { background: #dbeafe; align-self: flex-start; }
  .bot { background: #fff; border: 1px solid #e5e2da; }
  .meta { font-size: .8rem; color: #888; margin-top: 6px; }
  form { display: flex; gap: 8px; }
  input { flex: 1; padding: 12px; font-size: 1rem; border: 1px solid #ccc; border-radius: 10px; }
  button { padding: 12px 18px; font-size: 1rem; border: 0; border-radius: 10px; background: #1f6f5c; color: #fff; cursor: pointer; }
  button:disabled { opacity: .5; cursor: wait; }
</style>
</head>
<body>
<main>
  <h1>📚 مساعد القراءة</h1>
  <p class="sub">اسألني عن كتاب، وسأقترح عليك من كتالوجنا (٢٠٠ كتاب عربي).</p>
  <div id="log"></div>
  <form id="f">
    <input id="q" autocomplete="off" placeholder="مثال: أريد رواية تاريخية" required>
    <button id="b">اسأل</button>
  </form>
</main>
<script>
  const log = document.getElementById("log"), q = document.getElementById("q"), b = document.getElementById("b");
  function add(cls, text, meta) {
    const d = document.createElement("div");
    d.className = "msg " + cls;
    // Show **bold** from the model as real bold text (safely: text nodes, no innerHTML).
    text.split("**").forEach((part, i) => {
      const node = i % 2 ? document.createElement("strong") : document.createTextNode("");
      node.textContent = part;
      d.appendChild(node);
    });
    if (meta) { const m = document.createElement("div"); m.className = "meta"; m.textContent = meta; d.appendChild(m); }
    log.appendChild(d);
    d.scrollIntoView({ behavior: "smooth" });
    return d;
  }
  document.getElementById("f").addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = q.value.trim();
    if (!question) return;
    add("me", question);
    q.value = ""; b.disabled = true;
    const wait = add("bot", "… أبحث في الكتالوج");
    try {
      const r = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
      const data = await r.json();
      wait.remove();
      if (!r.ok) throw new Error(data.detail || "خطأ");
      add("bot", data.answer, data.searches.length ? "🔎 بحث: " + data.searches.join(" · ") : "");
    } catch (err) {
      wait.remove();
      add("bot", "⚠️ " + err.message);
    } finally {
      b.disabled = false; q.focus();
    }
  });
</script>
</body>
</html>
"""
