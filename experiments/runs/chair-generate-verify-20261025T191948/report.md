# Generate-and-verify with CHAIR labels — chair-generate-verify-20261025T191948

Labels = COCO instance annotations + objects named in the 5 reference captions (still incomplete); generator and verifier share SmolVLM weights.

- Images: 80; object-existence claims from generated captions: 141
- Raw captions: CHAIR-i = 0.079, CHAIR-s = 0.138 (139 mentions)
- After removing verifier-contradicted mentions: CHAIR-i = 0.059 (136 mentions kept)
- EAV-SA (default thresholds): coverage 0.901, accuracy on covered 0.984, hallucination precision 1.0, recall 0.6, effective recall 0.3
- Confusion (rows gold): {'SUPPORTED': {'SUPPORTED': 122, 'CONTRADICTED': 0, 'ABSTAIN': 9}, 'CONTRADICTED': {'SUPPORTED': 2, 'CONTRADICTED': 3, 'ABSTAIN': 5}}
- Runtime: 346.7 s
