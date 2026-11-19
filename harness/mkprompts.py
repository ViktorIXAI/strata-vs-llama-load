#!/usr/bin/env python3
"""Build reproducible long-context inputs from real corpora (docs + source code),
with unique control markers at ~2/26/51/76/98% depth. Exact token counts come from
the engine tokenizer (/tokenize), never from character estimates."""
import hashlib, json, os, random, sys, urllib.request

B = os.path.expanduser("~/bench-load")
HOST, PORT = "127.0.0.1", 18080
SEED = 20261118
DOCS = os.path.expanduser("~/basalt-041/docs")
CODE = os.path.expanduser("~/llama.cpp")


def corpus_files(kind, salt=0):
    out = []
    root = DOCS if kind == "docs" else CODE
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "build", "__pycache__", "third-party")]
        for fn in files:
            if kind == "docs" and fn.endswith(".md"):
                out.append(os.path.join(dirpath, fn))
            elif kind == "code" and fn.endswith((".cpp", ".h", ".py")):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    random.Random(SEED + 3 * salt).shuffle(out)
    return out


def read(path, cap=120_000):
    try:
        t = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""
    return t[:cap]


def markers(n=5, salt=0):
    rng = random.Random(SEED + 7 + 13 * salt)
    words = ["AZURE", "HARVESTER", "OBSIDIAN", "LANTERN", "MERIDIAN", "COPPER",
             "GLACIER", "THICKET", "QUARRY", "VESSEL", "ORCHARD", "BEACON"]
    out = []
    for i in range(n):
        a = rng.choice(words)
        b = rng.choice(words)
        out.append({"id": f"{a}-{b}-{rng.randint(1000,9999)}",
                    "value": f"{rng.randint(100,999)}.{rng.randint(10,99)}"})
    return out


def build(kind, target, tokenize, salt=0):
    """kind: chat | code | json. Returns (text, marker_list, tokens)."""
    mk = markers(5, salt) if target >= 65536 else []
    depth = [0.02, 0.26, 0.51, 0.76, 0.98][:len(mk)]
    files = corpus_files("docs" if kind == "chat" else "code", salt)
    if kind == "json":
        files = corpus_files("docs", salt) + corpus_files("code", salt)
    head = ("You are given a long reference dump. Every paragraph is real text from a project's "
            "documentation and source tree, concatenated in a fixed order. Answer the question at "
            "the very end using only this dump.\n\n")
    if mk:
        note = ("Control records are embedded in the dump; each is one line of the form "
                "CONTROL <id> VALUE <value>. Ignore them while reading, report them when asked.\n\n")
        head += note
    tail = {
        "chat": ("\n\nQuestion: compare how this project stores attention state and what it says "
                 "about decoding cost when that state does not fit the GPU. Quote the exact numbers "
                 "the dump gives, list the conditions, and explain the tradeoff in your own words."),
        "code": ("\n\nQuestion: one function in this dump computes a checksum over a buffer. Find it, "
                 "quote it exactly, explain what could make it read past the end, and propose a patch."),
        "json": ("\n\nAnswer with a single JSON object only, matching this schema: "
                 '{"items":[{"id":string,"category":string,"priority":integer,"value":number,'
                 '"comment":string}]}. Emit 40 items summarising the most important subsystems '
                 "mentioned in the dump; use numbers you can justify from the dump text.",
        )[0],
    }[kind]
    body, approx = [], 0.0
    inserted = 0
    for path in files:
        txt = read(path)
        if not txt:
            continue
        block = f"=== FILE: {path.replace(os.path.expanduser('~'), '$HOME')} ===\n{txt}\n\n"
        # place a marker roughly every depth step of the target size
        while mk and inserted < len(mk) and approx >= depth[inserted] * target:
            m = mk[inserted]
            block = f"CONTROL {m['id']} VALUE {m['value']}\n\n" + block
            inserted += 1
        body.append(block)
        approx += len(block) * 0.36          # only used to place markers
        if approx >= target * 1.05:
            break
    while mk and inserted < len(mk):
        m = mk[inserted]
        body.insert(len(body) // 2, f"CONTROL {m['id']} VALUE {m['value']}\n\n")
        inserted += 1
    text = head + "".join(body) + tail
    for _ in range(6):
        n = tokenize(text)
        if abs(n - target) <= target * 0.01:
            return text, mk, n
        if n < target:
            extra = ""
            for path in files[int(len(files) * 0.3):]:
                extra += read(path, 40_000)
                if len(extra) > (target - n) * 4:
                    break
            text += "\n\n" + extra
        else:
            cut = int(len(text) * (target / n))
            text = text[:cut] + tail
    return text, mk, tokenize(text)


def main():
    def tokenize(t):
        req = urllib.request.Request(f"http://{HOST}:{PORT}/tokenize",
                                     data=json.dumps({"content": t, "parse_special": False}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return len(json.loads(urllib.request.urlopen(req, timeout=180).read())["tokens"])

    want = [tuple(a.split(":")) for a in sys.argv[1:]]
    os.makedirs(B + "/prompts", exist_ok=True)
    man = {}
    for kind, ctx, rep in want:
        name = f"{kind}_{int(ctx)}_r{rep}"
        text, mk, n = build(kind, int(ctx), tokenize)
        p = f"{B}/prompts/{name}.txt"
        open(p, "w").write(text)
        man[name] = {"path": p, "tokens": n, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "markers": mk, "seed": SEED}
        print(json.dumps(man[name], ensure_ascii=False)[:400], flush=True)
    mf = f"{B}/prompts/manifest.json"
    old = json.load(open(mf)) if os.path.exists(mf) else {}
    old.update(man)
    json.dump(old, open(mf, "w"), indent=1)


if __name__ == "__main__":
    main()
