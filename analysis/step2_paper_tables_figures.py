# -*- coding: utf-8 -*-
"""
Step 2: tables and figures of the paper from the saved predictions of step 1 (no retraining; a few minutes).

Output: results/chronological/paper/ (figures/ as PDF and 600 dpi PNG, tables/Paper_Tables.xlsx with LaTeX and CSV
copies, Figure_Captions_Draft.md, README_paper.txt).

    python step2_paper_tables_figures.py
"""
import paper_outputs
from common import RESULTS_ROOT

N_BOOT = 2000                                 # bootstrap resamples over the test MC levels

if __name__ == "__main__":
    paper_outputs.build_paper_package(RESULTS_ROOT / "chronological", n_boot=N_BOOT)
