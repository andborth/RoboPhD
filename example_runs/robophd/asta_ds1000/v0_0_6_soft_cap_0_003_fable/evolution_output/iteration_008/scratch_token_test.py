import re, difflib

TOKEN_RE = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*")
NOISE_WORDS = {"array","dtype","tensor","object","name","length","freq",
               "int64","int32","int16","int8","uint8","uint16","uint32","uint64",
               "float64","float32","float16","bool","bool_","str_"}

def canon_num(tok):
    try: f = float(tok)
    except ValueError: return None
    if f != f or f in (float("inf"), float("-inf")): return None
    if f == int(f) and abs(f) < 1e15: return str(int(f))
    return "%.4g" % f

def cmp_tokens(text):
    out = []
    for m in TOKEN_RE.finditer(text or ""):
        tok = m.group(0)
        c = canon_num(tok)
        if c is not None: out.append(c); continue
        w = tok.lower()
        if w in NOISE_WORDS: continue
        out.append(w)
    return out

def containment(exp, got):
    sm = difflib.SequenceMatcher(None, exp, got, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / max(1, len(exp))

exp = cmp_tokens("array([7, 6, 3, 1, 3, 6, 3, 1])")
wrong = cmp_tokens("[8 6 3 1 3 6 3 1]")
right = cmp_tokens("[7 6 3 1 3 6 3 1]")
print("445 wrong:", containment(exp, wrong), "right:", containment(exp, right))

exp269 = cmp_tokens("""A_1,B_1,C_1,D_1,E_1,A_2,B_2_,C_2,D_2,E_2,A_3,B_3,C_3,D_3,E_3
1,2,3,4,5,6,7,8,9,10,11,12,13,14,5""")
wrong269 = cmp_tokens("   A_1  A_2  A_3  B_1  B_2  B_3  C_1  C_2  C_3  D_1  D_2  D_3  E_1  E_2  E_3\n0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15")
right269 = cmp_tokens("   A_1  B_1  C_1  D_1  E_1  A_2  B_2  C_2  D_2  E_2  A_3  B_3  C_3  D_3  E_3\n0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15")
print("269 wrong:", round(containment(exp269, wrong269),3), "right(w/ asker typos):", round(containment(exp269, right269),3))

exp238 = cmp_tokens("""   id city district         date  value
0   1   bj       ft  01-Jan-2019      1
1   2   bj       ft  01-Jan-2019      5
2   3   sh       hp  01-Feb-2019      1
3   3   sh       hp  01-Jan-2019      9
4   4   sh       hp  01-Feb-2019      5
5   4   sh       hp  01-Jan-2019     13
6   5   sh       hp  01-Feb-2019      9
7   5   sh       hp  01-Jan-2019     17
8   6  NaN      NaN  01-Feb-2019     13
9   7  NaN      NaN  01-Feb-2019     17""")
wrong238 = cmp_tokens("""   id city district         date  value
0   1   bj       ft  01-Jan-2019      1
1   2   bj       ft  01-Jan-2019      5
2   3   sh       hp  01-Jan-2019      9
3   3   sh       hp  01-Feb-2019      1
4   4   sh       hp  01-Jan-2019     13
5   4   sh       hp  01-Feb-2019      5
6   5   sh       hp  01-Jan-2019     17
7   5   sh       hp  01-Feb-2019      9
8   6  NaN      NaN  01-Feb-2019     13
9   7  NaN      NaN  01-Feb-2019     17""")
print("238 wrong:", round(containment(exp238, wrong238),3), "right:", round(containment(exp238, exp238),3))

# false-positive check: correct DataFrame rendered WITH index vs expected without index
expfp = cmp_tokens("birdType      birdCount\nAfrican Swallow          16510\nDead Parrot          16570\nExploding Penguin          16920")
gotfp = cmp_tokens("            birdType  birdCount\n0    African Swallow      16510\n1        Dead Parrot      16570\n2  Exploding Penguin      16920")
print("165-style correct w/ index:", round(containment(expfp, gotfp),3))
