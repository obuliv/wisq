# Test questions for the 3 sample documents

Based on the actual content of `Acme_Employee_Handbook_2025.docx`,
`Acme_Employee_Handbook_2026.docx`, and `APAC_Benefits_Handbook.docx`. Each
question is designed to exercise a specific piece of the retrieval/agentic
design, not just "does it find some text."

| # | Question | What it exercises | Expected-correct answer |
|---|---|---|---|
| 1 | How many days of paid time off do Acme employees get per year? | `is_latest` filtering / versioning — `search_documents` always filters `is_latest=True`, so this should surface the 2026 handbook, not the superseded 2025 one. | **15 days** (2026 handbook; 2025 said 14 and is superseded). |
| 2 | What was the PTO policy in the 2025 handbook, and how does it compare to the 2026 policy? | The hardest case: `search_documents` can *never* return a superseded document (`is_latest` is always baked in, not overridable). The only path to 2025's content is the model calling `get_related_documents` on the 2026 doc's `supersedes` row. Tests whether the two-tool design actually achieves "look up an older version" in practice. | 2025: 14 days/year. 2026: 15 days/year (increased by 1 day). |
| 3 | How many days of PTO do employees in Japan get? | Geography filter + cross-document precedence — the APAC handbook explicitly states its PTO section "TAKES PRECEDENCE" over the global handbook's for covered employees. | **12 days** (APAC Benefits Handbook), not 15 (global) — should cite the precedence rule, not just return the global number. |
| 4 | Are contractors based in China covered by the APAC Benefits Handbook? | A precise negative scope fact, not a precedence question — tests whether the scope statement is retrieved and read correctly rather than assumed. | **No** — the handbook explicitly excludes contractors ("It does NOT apply to contractors... Contractors... should refer to the global Acme Employee Handbook"). |
| 5 | What is the gym membership reimbursement for an employee based in Singapore? | Geography *exclusion* — the APAC handbook only covers China/Japan/Taiwan; Singapore isn't one of them, so this should fall back to the global policy, not APAC's. | **$50/month** (global handbook) — not $30 (APAC's rate, which doesn't apply to Singapore). |
| 6 | What is the gym membership reimbursement for an employee in Japan? Does it conflict with the global handbook's benefit, and if so, which one applies? | The subtlest case: PTO has an explicit APAC-wins rule, but gym membership doesn't — the *global* handbook's general precedence clause says "the MORE GENEROUS perk or benefit applies" for anything other than PTO. Tests whether the model over-generalizes "APAC always wins" vs. correctly applying the more-generous tiebreaker here. | **$50/month** (global) should apply, since it's more generous than APAC's $30 — the PTO-specific override doesn't extend to other benefits. |
| 7 | Does the regional APAC handbook or the global handbook govern PTO for an employee in Taiwan, and why? | Precedence direction/attribution with a "why" — checks whether the formatted `get_related_documents` explanation (and its underlying `source_text` quote) is actually surfaced in the answer, not just the winning number. | **APAC handbook**, specifically because it states its local PTO policy takes precedence over the global handbook's for covered employees. |
| 8 | Which Acme employee handbook version is currently in effect? | Simple `is_latest` sanity check, no precedence/geography involved. | **2026** (effective January 1, 2026); 2025 is superseded. |

## Known limitation this will likely expose

Question 6 is deliberately adversarial to the current `get_related_documents`
formatting: relationship rows extracted at ingestion carry a single `topic`
field (here, "PTO" specifically, since that's the only cue phrase the sample
text uses — "TAKES PRECEDENCE... with respect to PAID TIME OFF (PTO)"). The
global handbook's *general* precedence clause ("more generous perk or benefit
applies") is a separate, untopic'd rule that may or may not get extracted as
its own `DocumentRelationship` row depending on whether `SectionAnnotationExtractor`'s
cue-phrase gate catches it. Worth observing whether the answer to #6 is correct
by the model reasoning over the retrieved global-handbook text directly (which
states the rule in prose) even if no relationship row captures it — versus
getting it wrong by assuming the PTO precedence rule applies to all benefits.
