# GitHub publication

**Gut-RWKV** is published at `https://github.com/xiaol/Gut-RWKV` on the public
`main` branch. The repository contains a standalone install, tests, CI, browser
demo, reproducible training commands and public result summaries. The initial
published commit is `1f6028a` (`Introduce Gut-RWKV state-tuned decision model`). Base
weights, adapters, training data, `.venv`, local tools and `runs/` remain ignored.

For a future update, use the existing `origin` remote and push with:

```bash
git add .
git diff --cached --check
git commit -m "Describe public Gut-RWKV repository"
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git push origin main
```

Set the appropriate Git author identity before committing. Base weights, adapters,
training data, `.venv`, local tools and `runs/` remain ignored. A fresh clone can
reproduce the pilots; the locally trained adapters are not bundled in source.
The upstream runtime's MIT license is retained. The original Gut-RWKV code has
no selected root license yet; do not describe it as MIT/Apache without choosing one.

Do not claim an official JevBench rank, a state-tuning advantage over LoRA, or
production concurrency. The reports distinguish measured results from hypotheses.
