from inspect_ai.model import get_model
GPT_5_4_MINI = get_model("openai/gpt-5.4-mini")
try:
    GPT_5_4 = get_model("openai/gpt-5.4")
except Exception:
    GPT_5_4 = GPT_5_4_MINI
CLAUDE_SONNET_4_6 = GPT_5_4_MINI
