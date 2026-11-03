# -*- coding: utf-8 -*-
"""
Step 3: digital shadow evaluation on the archive, Section 2.3.2 of the paper (needs the checkpoints of step 1).

Baseline of the residual r = rho_g - rho_d, fault-free S4 monitoring, synthetic offsets in mass, attenuation and phase,
detection within five records and the effect of the offsets on predicted MC.
Output: results/digital_shadow/ (tables/, figures/, SUMMARY_digital_shadow.txt, manifest.json).

    python step3_digital_shadow_evaluation.py
"""
import shadow_evaluation

if __name__ == "__main__":
    shadow_evaluation.main()
