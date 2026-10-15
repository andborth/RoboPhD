import re
src = open("agent.py").read()

def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:100])
    src = src.replace(old, new)

# ---------------------------------------------------------------- docstring
rep('''"""ensemble-exec-vote-dtype-judge: DS-1000 solver (iteration 5).

Iteration 4's pipeline (three-provider ensemble, execute, majority vote, judge
on disagreement, generality check, loop guard, bounded waits) is kept verbatim.
New in this version:
  * dtype-split tie-break: when all runnable candidates agree on VALUES but
    differ in dtype/type (165: int counts vs the reference's column_stack
    strings), the majority vote is not trusted; an Opus judge reconstructs the
    accepted StackOverflow answer and picks the candidate whose dtypes match.
  * the ensemble is topped up with Sonnet whenever fewer than three candidates
    arrive (Gemini stalled on 2/20 samples), so the vote always has a majority.
  * generator / judge prompts mention dtype fidelity.
''', '''"""ensemble-exec-vote-escalate: DS-1000 solver (iteration 6).

Iteration 5's pipeline (three-provider ensemble, execute, majority vote, judge
on disagreement, dtype-split judge, generality check, loop guard, bounded
waits) is kept. New in this version:
  * the cost estimator no longer double-counts reasoning tokens for OpenAI /
    Anthropic handles (their output_tokens already include reasoning). In
    iteration 5 the inflated estimate tripped SOFT_CAP on two three-way splits
    and the disagreement judge was silently skipped (662, 918; the fallback
    happened to be right). The judge now has its own, higher cap.
  * escalation on disagreement: when no two runnable candidates agree on the
    example output, two more candidates (Claude Sonnet 4.6 medium, GPT-5.5)
    are generated in parallel and executed; a group of >=3 of 5 wins outright,
    otherwise the judge sees one representative per distinct output with the
    number of models that produced it.
  * the disagreement judge runs with a little reasoning (Opus 4.8, low).
  * Gemini's generation timeout is 150 s (it stalled again in iteration 5).
  * every candidate's strict / value-normalised output hash and dtype lines
    are logged so decisions can be replayed from agent_stdout.
''')

# ---------------------------------------------------------------- timeouts / caps
rep('''GEMINI_TIMEOUT = 240     # Gemini's successful calls take 5-50 s; it stalled for 330 s on 2/20 samples''',
    '''GEMINI_TIMEOUT = 150     # Gemini's successful calls take 5-62 s; it stalled for 240+ s on 3/40 samples''')
rep('''SOFT_CAP = 0.10  # skip optional LLM steps beyond this estimated spend per sample''',
    '''SOFT_CAP = 0.16    # skip optional LLM steps (repair/generality/loop rewrite/escalation) beyond this
JUDGE_CAP = 0.30   # the disagreement/dtype judges are the highest-value calls; only skip when far over''')

# ---------------------------------------------------------------- budget: no double counting
rep('''PRICES = {  # $/token (input, output)
    "gpt54": (2.50 / 1e6, 15.00 / 1e6),
    "opus": (5.00 / 1e6, 25.00 / 1e6),
    "gemini": (2.00 / 1e6, 12.00 / 1e6),
    "sonnet": (3.00 / 1e6, 15.00 / 1e6),
    "mini": (0.75 / 1e6, 4.50 / 1e6),
}''', '''PRICES = {  # $/token (input, output)
    "gpt54": (2.50 / 1e6, 15.00 / 1e6),
    "gpt55": (5.00 / 1e6, 30.00 / 1e6),
    "opus": (5.00 / 1e6, 25.00 / 1e6),
    "gemini": (2.00 / 1e6, 12.00 / 1e6),
    "sonnet": (3.00 / 1e6, 15.00 / 1e6),
    "mini": (0.75 / 1e6, 4.50 / 1e6),
}
# Verified against eval_cost in iteration 5: OpenAI and Anthropic usage.output_tokens already
# include reasoning tokens; Gemini bills reasoning on top of output_tokens.
REASONING_EXTRA = {"gemini"}''')
rep('''            r = getattr(u, "reasoning_tokens", 0) or 0
            c = i * pin + (o + r) * pout
            self.log.append(f"{key}: in={i} out={o} reas={r} ${c:.4f}")''',
    '''            r = getattr(u, "reasoning_tokens", 0) or 0
            c = i * pin + (o + (r if key in REASONING_EXTRA else 0)) * pout
            self.log.append(f"{key}: in={i} out={o} reas={r} ${c:.4f}")''')

# ---------------------------------------------------------------- imports
rep('''from model_registry import (
    CLAUDE_OPUS_4_8,
    CLAUDE_SONNET_4_6,
    GEMINI_3_1_PRO_PREVIEW,
    GPT_5_4,
    GPT_5_4_MINI,
)''', '''from model_registry import (
    CLAUDE_OPUS_4_8,
    CLAUDE_SONNET_4_6,
    GEMINI_3_1_PRO_PREVIEW,
    GPT_5_4,
    GPT_5_4_MINI,
    GPT_5_5,
)''')

# ---------------------------------------------------------------- judge prompt: group counts
rep('''    for c in cands:
        parts.append(f"\\n### Candidate {c['tag']}\\n<code>\\n{c['code']}\\n</code>\\n")
        if can_execute:
            parts.append(f"Execution report for {c['tag']}:\\n{c['run']['report'][:3500]}\\n")
    parts.append(JUDGE_RULES)
    return "".join(parts)''',
    '''    total = sum(c.get("n", 1) for c in cands)
    for c in cands:
        n = c.get("n", 1)
        hdr = f"\\n### Candidate {c['tag']}"
        if can_execute and total > len(cands):
            hdr += (f" (this output was produced by {n} of the {total} models; the other"
                    f" {n - 1} agreeing solution(s) are omitted)" if n > 1
                    else f" (this output was produced by 1 of the {total} models)")
        parts.append(f"{hdr}\\n<code>\\n{c['code']}\\n</code>\\n")
        if can_execute:
            parts.append(f"Execution report for {c['tag']}:\\n{c['run']['report'][:3500]}\\n")
    if can_execute and total > len(cands):
        parts.append("\\nAgreement between independent models is evidence but not proof: a minority "
                     "output is right when the majority shares a hard-coded assumption or ignores a "
                     "detail of the question. Weigh the counts together with the checks below.")
    parts.append(JUDGE_RULES)
    return "".join(parts)''')

# ---------------------------------------------------------------- escalation generators
rep('''def gen_timeout(key):
    return GEMINI_TIMEOUT if key == "gemini" else LLM_TIMEOUT
''', '''def gen_timeout(key):
    return GEMINI_TIMEOUT if key == "gemini" else LLM_TIMEOUT


# extra voices, generated only when the first three disagree pairwise
ESCALATION = [
    ("D", CLAUDE_SONNET_4_6, "sonnet", "medium", 3500, 4),
    ("E", GPT_5_5, "gpt55", None, 6000, 5),
]


def out_hash(s: str) -> str:
    import hashlib
    return hashlib.md5((s or "").encode("utf-8", "replace")).hexdigest()[:8]


def dtype_lines(out: str) -> str:
    d = re.findall(r"^(?:\\[dtypes\\].*|type=.*|.*\\[dtype\\] .*)$", out or "", re.M)
    return " | ".join(x.strip() for x in d)[:300]
''')

# ---------------------------------------------------------------- exec logging
rep('''    for c in cands:
        c["run"] = await execute(c["code"])
        print(f"  [t={ELAPSED():.0f}s] exec {c['tag']} ok={c['run']['ok']}\\n{c['run']['report'][:1200]}")
    can_execute''', '''    for c in cands:
        c["run"] = await execute(c["code"])
        print(f"  [t={ELAPSED():.0f}s] exec {c['tag']} ok={c['run']['ok']}\\n{c['run']['report'][:1200]}")
        if c["run"]["ok"]:
            print(f"  keys {c['tag']}: strict={out_hash(norm_out(c['run']['out']))} "
                  f"values={out_hash(norm_values(c['run']['out']))} dtypes: {dtype_lines(c['run']['out'])}")
    can_execute''')

# ---------------------------------------------------------------- vote: escalation + grouping
rep('''    chosen = None
    majority_grp = None
    if can_execute:
        good = [c for c in cands if c["run"]["ok"]]
        groups = {}
        for c in good:
            groups.setdefault(norm_out(c["run"]["out"]), []).append(c)
        sizes = sorted((len(v) for v in groups.values()), reverse=True)
        print(f"  runnable={len(good)}/{len(cands)} agreement groups={sizes}")
        if sizes and sizes[0] >= 2 and (len(sizes) == 1 or sizes[0] > sizes[1]):
            grp = max(groups.values(), key=len)
            chosen = sorted(grp, key=rank_key)[0]
            majority_grp = grp
            print(f"  majority of {len(grp)} agrees -> candidate {chosen['tag']}")
        pool = good if good else cands
''', '''    chosen = None
    majority_grp = None
    if can_execute:
        def regroup():
            good = [c for c in cands if c["run"]["ok"]]
            groups = {}
            for c in good:
                groups.setdefault(norm_out(c["run"]["out"]), []).append(c)
            sizes = sorted((len(v) for v in groups.values()), reverse=True)
            print(f"  runnable={len(good)}/{len(cands)} agreement groups={sizes}")
            return good, groups, sizes

        good, groups, sizes = regroup()
        # no two candidates agree -> escalate: two more voices before judging
        if (len(good) >= 2 and sizes[0] == 1 and budget.spent < SOFT_CAP and ELAPSED() < OPTIONAL_AFTER
                and not sb.dead):
            print(f"  [t={ELAPSED():.0f}s] no agreement among {len(good)} candidates -> escalating with "
                  f"{', '.join(k for (_t, _m, k, _e, _mt, _p) in ESCALATION)}")

            async def egen(tag, model, key, effort, mt):
                try:
                    txt = await llm(model, key, msgs, budget, effort=effort, max_tokens=mt, timeout=240)
                    return fix_indent(extract_code(txt), in_function)
                except Exception as e:
                    print(f"  escalation generation {tag} failed: {e!r}")
                    return ""

            eraw = await asyncio.gather(*[egen(t, m, k, e, mt) for (t, m, k, e, mt, _p) in ESCALATION])
            for (tag, _m, key, _e, _mt, prio), code in zip(ESCALATION, eraw):
                if not code.strip():
                    continue
                c = {"tag": tag, "key": key, "prio": prio, "code": code, "run": None, "run2": None}
                c["run"] = await execute(code)
                print(f"  candidate {tag} ({key}):\\n{code}")
                print(f"  [t={ELAPSED():.0f}s] exec {tag} ok={c['run']['ok']}\\n{c['run']['report'][:1200]}")
                if c["run"]["ok"]:
                    print(f"  keys {tag}: strict={out_hash(norm_out(c['run']['out']))} "
                          f"values={out_hash(norm_values(c['run']['out']))} dtypes: {dtype_lines(c['run']['out'])}")
                cands.append(c)
            good, groups, sizes = regroup()
        need = 2 if len(good) <= 3 else 3
        if sizes and sizes[0] >= need and (len(sizes) == 1 or sizes[0] > sizes[1]):
            grp = max(groups.values(), key=len)
            chosen = sorted(grp, key=rank_key)[0]
            majority_grp = grp
            print(f"  majority of {len(grp)} (of {len(good)} runnable) agrees -> candidate {chosen['tag']}")
        elif sizes and sizes[0] >= 2:
            print(f"  largest group {sizes[0]} of {len(good)} is not decisive; judging")
        if good and len(groups) >= 2:
            # one representative per distinct output; the judge sees the counts
            pool = []
            for g in groups.values():
                r = sorted(g, key=rank_key)[0]
                r["n"] = len(g)
                pool.append(r)
            pool.sort(key=lambda c: (-c["n"], rank_key(c)))
        else:
            pool = good if good else cands
''')

# ---------------------------------------------------------------- judge: low reasoning, own cap
rep('''        try:
            jv = await llm(CLAUDE_OPUS_4_8, "opus",
                           [ChatMessageSystem(content=SYSTEM_PROMPT), ChatMessageUser(content=prompt_text)],
                           budget, max_tokens=4000)
        except Exception as e:
            print(f"  judge failed: {e!r}")
            return None''', '''        try:
            jv = await llm(CLAUDE_OPUS_4_8, "opus",
                           [ChatMessageSystem(content=SYSTEM_PROMPT), ChatMessageUser(content=prompt_text)],
                           budget, effort="low", max_tokens=4000)
        except Exception as e:
            print(f"  judge failed: {e!r}")
            return None''')
rep('''    if chosen is None and budget.spent < SOFT_CAP and ELAPSED() < JUDGE_AFTER:
        chosen = await judge(judge_prompt(problem, pool, can_execute))''',
    '''    if chosen is None and budget.spent < JUDGE_CAP and ELAPSED() < JUDGE_AFTER:
        chosen = await judge(judge_prompt(problem, pool, can_execute))
    elif chosen is None:
        print(f"  judge skipped (est_cost=${budget.spent:.3f}, t={ELAPSED():.0f}s)")''')
rep('''    dsplit = ctx.get("dtype_split")
    if dsplit and budget.spent < SOFT_CAP and ELAPSED() < JUDGE_AFTER:''',
    '''    dsplit = ctx.get("dtype_split")
    if dsplit and budget.spent < JUDGE_CAP and ELAPSED() < JUDGE_AFTER:''')
rep('''                    if budget.spent < SOFT_CAP and ELAPSED() < JUDGE_AFTER:
                        j = await judge(general_judge_prompt(problem, ok2, skeleton2), ref_out=ref_out)''',
    '''                    if budget.spent < JUDGE_CAP and ELAPSED() < JUDGE_AFTER:
                        j = await judge(general_judge_prompt(problem, ok2, skeleton2), ref_out=ref_out)''')

# the judge's CHOOSE lookup must also see candidates that are not in the (deduplicated) pool
rep('''        if m:
            tag = m.group(1).upper()
            for c in pool:
                if c["tag"] == tag:
                    print(f"  judge chose candidate {tag}")
                    return c
        print(f"  judge output unusable: {jv[:200]!r}")
        return None''', '''        if m:
            tag = m.group(1).upper()
            for c in list(pool) + cands:
                if c["tag"] == tag:
                    print(f"  judge chose candidate {tag}")
                    return c
        print(f"  judge output unusable: {jv[:200]!r}")
        return None''')

# emergency fallback: also log budget key for GPT_5_5 if used (n/a) -- nothing

open("agent.py", "w").write(src)
print("patched OK", len(src.splitlines()), "lines")
