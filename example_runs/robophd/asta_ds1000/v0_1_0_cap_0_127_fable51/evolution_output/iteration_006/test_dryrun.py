import sys, os, types, asyncio, io, contextlib, re
sys.path.insert(0, os.getcwd())

class FakeUsage:
    input_tokens = 1500; output_tokens = 400; reasoning_tokens = 800
class FakeResp:
    def __init__(self, text): self.completion = text; self.usage = FakeUsage()
CANNED = {}
class FakeModel:
    def __init__(self, name): self.name = name; self.calls = []
    async def generate(self, messages, config=None):
        self.calls.append((messages, config))
        if os.environ.get("STALL") == self.name:
            await asyncio.sleep(3600)
        last = messages[-1].content if isinstance(messages, list) else messages
        if "they DIVERGE" in last:
            return FakeResp(CANNED.get("gjudge", "CHOOSE: A"))
        if "SAME VALUES" in last:
            if os.environ.get("FAIL") == "dtype":
                raise RuntimeError("provider down")
            return FakeResp(CANNED.get("dtype", "CHOOSE: A"))
        if "Several candidate solutions" in last:
            return FakeResp(CANNED.get("judge", "CHOOSE: A"))
        if "CRASHED" in last:
            return FakeResp(CANNED.get("repair", "<code>\nresult = None\n</code>"))
        if "contains the token `for`" in last:
            return FakeResp(CANNED.get("loop", "<code>\nresult = 1\n</code>"))
        if "GENERALITY" in last:
            return FakeResp(CANNED.get("alt", "<code>\n" + CANNED["_skeleton"] + "\n</code>"))
        return FakeResp(CANNED[self.name])
mr = types.ModuleType("model_registry")
for n in ["CLAUDE_OPUS_4_8", "GPT_5_4", "GEMINI_3_1_PRO_PREVIEW", "CLAUDE_SONNET_4_6", "GPT_5_4_MINI", "GPT_5_5"]:
    setattr(mr, n, FakeModel(n))
sys.modules["model_registry"] = mr

from inspect_ai.tool import tool
from inspect_ai.solver import TaskState
from inspect_ai.model import ModelOutput

@tool
def python_session():
    async def execute(code: str) -> str:
        """Run python code.

        Args:
            code: the code
        """
        if os.environ.get("STALL") == "sandbox":
            await asyncio.sleep(3600)
        buf = io.StringIO(); ns = {}
        with contextlib.redirect_stdout(buf):
            try:
                exec(code, ns)
            except Exception as e:
                print("ERR", repr(e))
        return buf.getvalue()
    return execute

import agent
for k in ("LLM_TIMEOUT", "GEMINI_TIMEOUT", "SANDBOX_TIMEOUT", "PROBE_TIMEOUT", "HARD_LIMIT", "OPTIONAL_AFTER", "JUDGE_AFTER"):
    if os.environ.get(k):
        setattr(agent, k, float(os.environ[k]))

def ref_code(pid, base):
    s = open(f"{base}/{pid}/reference.md").read()
    m = re.search(r"```(?:python)?\n(.*?)```", s, re.S)
    return "<code>\n" + (m.group(1) if m else s) + "\n</code>"

async def main(pid):
    base2 = "../../iteration_002/agent_iter2_exec_verified_sonnet/problems"
    baseb = "../../iteration_002/agent_baseline/problems"
    if not os.path.exists(f"{base2}/{pid}"):
        base2 = "../../iteration_003/agent_iter2_exec_verified_sonnet/problems"
        baseb = "../../iteration_003/agent_baseline/problems"
    prob = open(f"{base2}/{pid}/problem.md").read()
    lib = open(f"{base2}/{pid}/library").read().strip()
    CANNED["CLAUDE_OPUS_4_8"] = open(f"{base2}/{pid}/extracted_code.md").read()
    CANNED["GPT_5_4"] = open(f"{baseb}/{pid}/extracted_code.md").read()
    CANNED["GEMINI_3_1_PRO_PREVIEW"] = ref_code(pid, base2)
    CANNED["CLAUDE_SONNET_4_6"] = CANNED["GEMINI_3_1_PRO_PREVIEW"]
    CANNED["GPT_5_4_MINI"] = CANNED["GPT_5_4"]
    CANNED["GPT_5_5"] = CANNED["GPT_5_4"] if os.environ.get("E_AS") == "B" else CANNED["GEMINI_3_1_PRO_PREVIEW"]
    if os.environ.get("D_AS") == "A": CANNED["CLAUDE_SONNET_4_6"] = CANNED["CLAUDE_OPUS_4_8"]
    if os.environ.get("D_AS") == "B": CANNED["CLAUDE_SONNET_4_6"] = CANNED["GPT_5_4"]
    CANNED["_skeleton"] = agent.parse_prompt(prob)[0]
    if os.environ.get("LOOPTEST"):
        CANNED["GPT_5_4"] = CANNED["GEMINI_3_1_PRO_PREVIEW"] = CANNED["CLAUDE_OPUS_4_8"]
    TAGMAP = {"A": "CLAUDE_OPUS_4_8", "B": "GPT_5_4", "C": "GEMINI_3_1_PRO_PREVIEW", "D": "CLAUDE_SONNET_4_6", "E": "GPT_5_5"}
    for a in sys.argv[2:]:
        if len(a) > 2 and a[0] in TAGMAP and a[1] == "=":
            CANNED[TAGMAP[a[0]]] = "<code>\n" + a[2:] + "\n</code>"
        for k in ("judge", "gjudge", "loop", "repair", "alt", "dtype"):
            if a.startswith(k + "="):
                v = a[len(k) + 1:]
                CANNED[k] = open(v).read() if os.path.exists(v) else v
    st = TaskState(model="fake/model", sample_id=pid, epoch=0, input=prob, messages=[], metadata={"library": lib})
    st.tools = [python_session()]
    st.output = ModelOutput()
    st = await agent.make_solver()(st, None)
    print("COMPLETION>>>", st.output.completion)

asyncio.run(main(sys.argv[1]))
