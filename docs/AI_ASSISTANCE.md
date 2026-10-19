# AI assistance log

Tool: Claude Code (Anthropic, Claude Opus model) in VS Code. Claude Code wrote most of the code. Each prompt I gave it is rewritten below in STAR form (Situation, Task, Action, Result). Credentials are omitted.

## 1. Read the brief
- **Situation:** I received the assessment document and three CSVs, with a short deadline.
- **Task:** Understand the requirements before building anything.
- **Action:** Asked Claude Code to read the assessment and summarise its guidelines.
- **Result:** A summary of the agent contract, labeling rules, harness metrics, run sequence, API/DB and test requirements. Environment blockers found: no API key, no PostgreSQL.

## 2. Provision the environment
- **Situation:** No LLM key or database was configured locally.
- **Task:** Install the dependencies and use my Groq API key.
- **Action:** Installed pinned Python packages and an embedded PostgreSQL 16 (`pgserver`, no admin rights needed). Stored the key only in a git-ignored `.env`. Built an OpenAI-compatible provider adapter so Groq, OpenAI and Gemini are interchangeable.
- **Result:** Key verified live. Rate limits found: 8,000 tokens/min on the free tier. The deviation from the OpenAI/Anthropic/Gemini list is disclosed in the README.

## 3. Set quality expectations
- **Situation:** The brief does not mark a frontend (React is optional, no marks).
- **Task:** Confirm frontend scope, keep the work to industry standard, and log my prompts in STAR format.
- **Action:** Skipped the frontend. Applied typed validation, parameterized SQL, DB constraints, sanitized errors, pinned dependencies, secret hygiene and controlled tests.
- **Result:** This log and the README.

## 4 to 6. Plan for time pressure
- **Situation:** I wanted the assessment finished in 10, then 5, minutes.
- **Task:** Estimate the remaining effort and cut it down.
- **Action:** Claude Code estimated about 15 minutes of coding plus about 1.5 hours of live runs, limited by Groq's 8k tokens/min. It started the v1 run in the background and built the tests and API in parallel.
- **Result:** Code and tests finished. The live runs are rate-limited and still in progress.

## 7. Check whether live runs are required
- **Situation:** Live runs are slow on the free tier.
- **Task:** Check whether the brief requires them.
- **Action:** Re-read the brief: "Submit four actual live-run artifacts" and "Fakes do not replace the four live runs".
- **Result:** The runs are required. They were kept running, and anything incomplete is stated as unfinished work.

## 8. Submit at the deadline
- **Situation:** The submission deadline was 17:00 IST.
- **Task:** Submit on time with honest status.
- **Action:** Followed the brief's rule "Stop at the limit and state unfinished work". Wrote the README status table and committed.
- **Result:** The submission records what is done, what is in progress and what is unfinished.

## 9. Design the v2 change
- **Situation:** v1 was producing results, and the brief requires one targeted change based on v1's errors.
- **Task:** Write the v2 prompt from the v1 error evidence and keep a clean single-change comparison.
- **Action:** Scored v1's first 11 development answers with the harness, then grouped the errors: 4 missed `unknown` claims and 2 wrong labels. Confirmed from the A10 trace that the model dropped claims the catalogue couldn't verify. Added a single coverage rule to make v2. Kept STAR formatting to this log, because restructuring the prompt itself would have changed many behaviors at once.
- **Result:** `app/prompts/v2.md` differs from v1 by one line, and all 52 tests pass. The v2 runs are queued after v1.

## 10. Draft the submission email
- **Situation:** The work had to be submitted with the remaining runs still pending.
- **Task:** Draft a short submission email for the hiring team.
- **Action:** Wrote an email giving the repository link, what is included, the honest status with unfinished work, the Groq deviation and the AI-assistance disclosure.
- **Result:** An email ready to send, with placeholders for names.

## Checks performed during the build (automated and observed)
- 52 controlled tests were written and actually run, and all pass; the output is in `reports/test_output.txt`.
- Every one of the 376 gold offsets slices the loaded `ai_answer` exactly. This shows CSV decoding is lossless.
- Live smoke tests on A01, A02 and A03: valid spans every time, and 10 of 10 exact span matches on A02.
- Found and fixed a real provider failure: the undeclared `json` tool call (see README).

## What I verified myself
_(Candidate to complete: list the files you read, the commands you ran and the outputs you checked yourself.)_
