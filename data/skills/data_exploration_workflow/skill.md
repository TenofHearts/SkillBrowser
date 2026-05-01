# Data Exploration Workflow

## Overview

Use this workflow before modeling, charting, or making decisions from tabular data. The goal is to understand what the data contains, where it is unreliable, and which patterns deserve follow-up.

## Procedure

1. Inspect rows, columns, data types, units, and identifiers.
2. Check missing values, duplicates, impossible values, and category inconsistencies.
3. Summarize numeric distributions and categorical frequencies.
4. Compare important groups and time periods when present.
5. Flag outliers and decide whether they are errors or meaningful cases.
6. Separate observed patterns from causal claims.

## Output Format

Return markdown with:

- `Dataset Shape`
- `Column Notes`
- `Data Quality`
- `Patterns`
- `Recommended Next Steps`
- `Caveats`

## Examples

For "Explore this sales CSV", report date coverage, revenue distribution, missing fields, top categories, and suspicious records.

## Failure Modes

Do not infer causality from correlations. Do not ignore missingness or duplicate records when reporting trends.
