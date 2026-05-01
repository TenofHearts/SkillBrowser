# CSV Reader

## Overview

Use this executable metadata when a local CSV file must be loaded before profiling, cleaning, charting, or analysis.

## Schema

Input requires `file_path`. Optional fields include `delimiter`, `encoding`, and `max_rows`.

Output includes `columns`, `row_count`, and an optional `sample` of records.

## Examples

Invoke before data exploration tasks such as "show me the columns", "profile this dataset", or "find missing values in this CSV".

## Failure Modes

Incorrect delimiters or encodings can produce malformed columns. Very large files may need a future chunked reader.
