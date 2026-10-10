# Demo video script (target 4:30, hard limit 5:00)

Requirements from the rules: public, viewable without login, shows real inputs, workflow, outputs and results; every image/music/third-party item must have redistribution rights. Chip photos are CC-BY (credit on screen); use no music or use royalty-free.

| Time | Screen | Say (plain words) |
|---|---|---|
| 0:00–0:25 | Title card; category "Model & Algorithm" | Organ-on-chip experiments make many photos; some are unusable and finding them by eye is slow. We built a review aid that puts the worst photos first, and we tested it without shortcuts. |
| 0:25–1:10 | Terminal: `python inference.py --photos demo_photos --out ranking.csv --sheet sheet.png`; then `ranking.csv` and `demo_sheet.png` (from the demo job) | Input: a folder of photos. Output: a ranked list. The top row is reviewed first. Photos in a camera format we never validated are not scored; they are sent to a person. |
| 1:10–2:10 | Slide: final-test table (AUC 0.850 vs 0.80–0.81; +0.04 to +0.05) and the catch curve (0.703 vs random 0.499) | On photos locked away before any training, zoomed tiles beat the plain model, and reviewing in our order finds bad photos far faster than random. |
| 2:10–2:50 | Slide: "what we withdrew" — unseen-cell-type claim (0.86 → 0.76 → X01 0.80), light retraining worse, heat maps unreliable, tweaks no gain | We also report what failed. When we kept whole filename groups out of training, scores dropped, so we withdrew our first claim. |
| 2:50–3:40 | Slides: nuclei — outlines 0.90 vs 0.86; noise failure and the fix (0.07 → 0.87, synthetic); tiny nuclei 43%; colour images fail | Same care for nucleus counting and outlining: clear outline gain, small counting gain, strong noise breaks it unless trained for it, tiny nuclei and colour images stay limits. |
| 3:40–4:10 | Slide: clean rebuild (42 min, fresh environment, free GPU) reproduces 0.8751 / 0.8496 | Everything reproduces from public data on a free GPU: the rebuild matches our records to four decimals. |
| 4:10–4:30 | Limits slide: simulated workload not time saved; one dataset; small format unvalidated; next steps (verified chips, timed review study) | Limits: the review curves are simulated; we have not measured time saved. Next: verified physical chips and a timed study with real reviewers. |

Assets are ready in `code/demo/` (`demo_ranking_with_labels.csv`, `demo_sheet.png`, `NOTES.md`): 12 random locked-final photos ranked (the 6 bad ones took ranks 1–6) and 5 photos sent to manual review. Say on screen that this is a 12-photo illustration and that the accuracy evidence is the final-test table. Other assets (from the stored runs): the catch-curve CSV (`runs/*F01_chip_final_eval/catch_curves.csv`), the results table (file 16). Credit line on screen: "Organ-on-a-Chip image dataset, Zenodo 10203721, CC-BY 4.0".
