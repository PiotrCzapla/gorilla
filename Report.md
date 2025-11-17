# BFCL v4 Multi-Turn Runner

## What was built
- Added OpenRouter-aware model configs (`anthropic/claude-haiku-4.5` prompt + FC) so BFCL can talk to the Anthropic Haiku 4.5 SKUs through the OpenAI-compatible chat-completions endpoint.
- Created `run-multi.py`, a thin orchestration layer around the BFCL generation/evaluation modules. It sets up all OpenRouter headers, pins `BFCL_PROJECT_ROOT` to your chosen report directory, and records metadata about every execution (including logs, score files, and failure summaries) so you can diff FC vs. prompt behavior.
- The runner automatically creates `analysis/*.jsonl` files that list the test IDs each model missed and a `comparison.json` file that highlights which IDs are unique failures for each variant.

## Running the tool
1. **Install dependencies** (from repo root):
   ```bash
   python -m pip install -e berkeley-function-call-leaderboard
   ```
2. **Expose credentials**: ensure `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) is set. The script automatically maps it to OpenAI's SDK and injects the `HTTP-Referer` + `X-Title` headers demanded by OpenRouter.
3. **Mini smoke test** (first 5 multi-turn cases, FC + prompt):
   ```bash
   ./run-multi.py --mini -d ./report-dir/haiku -m anthropic/claude-haiku-4.5
   ```
   This writes:
   - `run_history.jsonl` – timestamped summary of each mode
   - `result/<model>/...json` – raw transcripts + tool traces
   - `score/<model>/...json` – JSONL scores for `multi_turn_base`
   - `analysis/*` – `*_failures.jsonl` plus `comparison.json`
   - `logs/*.log` – stdout for the generation/eval scripts
4. **Full benchmark** (multi-turn base across all entries):
   ```bash
   ./run-multi.py -d ./report-dir/haiku -m anthropic/claude-haiku-4.5
   ```
   Add `--mode fc` or `--mode prompt` if you need to isolate a single variant, adjust `--threads` for API parallelism, and switch `--category` if you want to probe other multi-turn subsets.

The script keeps the BFCL outputs in the same directory tree (`result/`, `score/`, `analysis/`), so you can immediately `grep` or `jq` through the JSON files to inspect behavior. Because every run also appends to `run_history.jsonl`, you can reconstruct the chronology of experiments (model ID, mini/full flag, score snapshot, etc.).

## Current state of executions
The committed `report-dir/haiku` folder contains the smoke-test artifacts from `./run-multi.py --mini ...` on 2025‑11‑17. Both FC and prompt attempts hit the OpenRouter hard limit immediately – every request returned `403 Key limit exceeded (monthly limit)`. You can verify this in:
- `analysis/fc_failures.jsonl`
- `result/anthropic_claude-haiku-4.5-FC/..._result.json`
- `logs/generate_*.log`
- `run_history.jsonl`

Because the key is capped, we could not finish either the smoke test nor the full benchmark, so no valid accuracy numbers or detailed failure conversations are available yet. Once OpenRouter lifts the limit (or a working Anthropic key is supplied), re-running the same command will populate the `analysis` directory with real successes/failures and the `comparison.json` file will highlight model-specific misses. The runner will also capture complete transcripts for your requested set of failed cases so you can write the FC vs. prompt analysis section the user asked for.

## How to inspect failures and compare models
- `analysis/<mode>_failures.jsonl` – newline-delimited JSON with `{id, status, reason}` for every miss or exception.
- `analysis/comparison.json` – merged view of FC vs. prompt misses. Any ID present in only one section indicates a unique failure for that variant.
- `result/<model>/..._result.json` – per-case transcripts with the raw tool invocations and checker verdicts.
- `logs/generate_*.log` / `logs/evaluate_*.log` – raw CLI output for auditing.

These files are regular text, so you can `grep` or `jq` them directly. Example:
```bash
jq 'select(.status=="incorrect")' report-dir/haiku/analysis/fc_failures.jsonl
```
will show every FC miss along with the stored reason.

## Outstanding work / next steps
1. Re-run `./run-multi.py --mini ...` once the OpenRouter key has quota again to verify the FC accuracy trends on the subset.
2. Launch the full benchmark (`./run-multi.py -d ./report-dir/haiku -m anthropic/claude-haiku-4.5`) and confirm the FC score is ~62 on `multi_turn_base`; repeat for prompt mode.
3. Use the refreshed `analysis/*.jsonl` files to pull at least 5 real FC failures and 5 prompt failures into the "failure case study" section of this report.
4. Commit the new `result/` and `score/` folders so the traces can be reviewed offline.

Until the API limit is resolved the evaluation cannot progress, but all infrastructure required to hit the target benchmark is now in place.
