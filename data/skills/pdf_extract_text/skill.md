# PDF Text Extractor

## Overview

Use this executable metadata when a local PDF must be converted into page-ordered text before analysis. It is suitable for digital PDFs with embedded text.

## Schema

Input requires `file_path`. Optional `pages` limits extraction to one-based page numbers.

Output returns `pages`, where each item contains `page_number` and `text`.

## Examples

Invoke for tasks such as summarizing a PDF paper, extracting a method section, or searching a local report for a term.

## Failure Modes

Scanned image PDFs may return little or no text. Use an OCR skill instead when the PDF has no embedded text layer.
