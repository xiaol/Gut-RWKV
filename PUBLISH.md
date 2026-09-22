# GitHub publication

The repository is prepared locally as **Gut-RWKV**, with a standalone install,
tests, CI, browser demo, reproducible training commands and public result summaries.
It has not been published: the environment has no authenticated GitHub API
credential or GitHub CLI session. Existing checkouts point at `xiaol`, but the
intended owner has not been confirmed. No credentials should be committed.

After signing in to GitHub CLI on this machine and choosing the owner:

```bash
gh auth login
gh auth status
git add .github .gitignore README.md RESEARCH.md RESULTS.md EXPERIMENTS.md SOTA.md \
  THIRD_PARTY.md PUBLISH.md pyproject.toml examples reports rwkv_jev scripts tests
git diff --cached --check
git diff --cached --stat
git commit -m "Introduce Gut-RWKV state-tuned decision model and reproducible pilots"
gh repo create OWNER/Gut-RWKV --public --source=. --remote=origin \
  --description "Typed decisions from RWKV recurrent memory: state tuning, batched branches, and reproducible evaluations" --push
```

Set the appropriate Git author identity before committing. Base weights, adapters,
training data, `.venv`, local tools and `runs/` remain ignored. A fresh clone can
reproduce the pilots; the locally trained adapters are not bundled in source.
The upstream runtime's MIT license is retained. The original Gut-RWKV code has
no selected root license yet; do not describe it as MIT/Apache without choosing one.

Do not claim an official JevBench rank, a state-tuning advantage over LoRA, or
production concurrency. The reports distinguish measured results from hypotheses.
