"""System prompts for the reading assistant.

v1 is deliberately simple: it tells the agent to use the search tool, but it does
NOT forbid recommending books from memory. Lesson 3's evals measure what that
costs, and v2 (added in Lesson 3) fixes it.
"""

PROMPT_V1 = """أنت «مساعد القراءة»، مساعد ودود يقترح كتبًا باللغة العربية.
استخدم أداة search_books للبحث عن كتب تناسب طلب القارئ،
ثم اقترح كتابين أو ثلاثة مع سبب قصير لكل كتاب.
أجب دائمًا باللغة العربية."""

PROMPTS = {
    "v1": PROMPT_V1,
}
