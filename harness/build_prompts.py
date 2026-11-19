#!/usr/bin/env python3
"""Build the prompt set for the load bench. Token targets are estimated here
(chars/4); the actual token counts are recorded per run by the engine /tokenize."""
import hashlib, json, os, sys

B = os.path.expanduser("~/bench-load")
sys.path.insert(0, B)
import mkprompts as MP

SPEC = [("p4k", "chat", 4096, 0), ("p4k_a", "chat", 4096, 11), ("p4k_f1", "chat", 2000, 7),
        ("p32k", "chat", 32768, 1), ("p32k_a", "chat", 32768, 21),
        ("p32k_b", "chat", 32768, 22), ("p32k_c", "chat", 32768, 23),
        ("p40k", "chat", 40960, 3), ("p128k", "json", 131072, 4),
        ("p64k_code", "code", 65536, 6), ("p184k", "json", 155648, 5)]
est = lambda t: max(1, len(t) // 4)

os.makedirs(B + "/prompts", exist_ok=True)
man = {}
for name, kind, target, salt in SPEC:
    text, mk, n = MP.build(kind, target, est, salt=salt)
    p = f"{B}/prompts/{name}.txt"
    open(p, "w").write(text)
    man[name] = {"path": p, "target_tokens": target, "est_tokens": est(text),
                 "kind": kind, "salt": salt, "seed": MP.SEED, "markers": mk,
                 "sha256": hashlib.sha256(text.encode()).hexdigest()}
    print(f"built {name}: target={target} est={man[name]['est_tokens']} markers={len(mk)}")
json.dump(man, open(B + "/prompts/manifest.json", "w"), indent=1)
