# Paper Claim-Method-Finding Workflow

## Overview

Use this workflow when a task needs more than a generic paper summary. The goal is to separate what the paper claims, how it supports the claim, what it found, and where the evidence is weak.

## Procedure

1. Identify the paper's research question or problem.
2. Extract the central claim or contribution in one sentence.
3. List the method: model, experiment, dataset, proof, survey, or qualitative approach.
4. Record the main findings and connect each finding to its evidence.
5. Note limitations, assumptions, missing comparisons, and threats to validity.
6. Keep speculation separate from what the paper explicitly states.

## Output Format

Return concise markdown with these headings:

- `Claim`
- `Method`
- `Findings`
- `Evidence`
- `Limitations`
- `Open Questions`

## Examples

For "What is the main contribution of this paper?", answer with the claim first, then add method and evidence only as support.

## Failure Modes

Do not treat background motivation as the central claim. Do not invent results when the text only describes a proposed method.
