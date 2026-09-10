# 交付检查

- pdf: C1C2机制重构研究与筛查_20260906.pdf
- pages: 10
- text_extract: pdftotext -layout: all 10 pages nonempty, Chinese text present
- visual_check: All 10 pages rendered by pdftoppm at scale-to 1250 and inspected with view_image; tables, two diagrams, body text and citations legible; no clipping observed
- numeric_check: scripts/summarize_evidence.py re-derived main comparison rows and weighted baselines; passed
- source_check: 17 report references mapped to primary-source ledger with access limitations
- current_boundary: All new metrics remain CPU-level; no new RTL, ASIC PPA, or network AEE result
