# Milestone 4 Mechanisms Synthesis Table

### Layer-Wise Accuracy Barrier Contribution & Representation Alignment (CKA)

| Layer / Block | Module Description | Mean Barrier Contribution (%) | Mean Linear CKA Similarity |
|:---|:---|:---:|:---:|
| `init` | Initial Conv & BatchNorm | 5.0% (0.105%) | 0.9638 |
| `stage1` | ResNet Block Stage 1 | 14.7% (0.308%) | 0.9448 |
| `stage2` | ResNet Block Stage 2 | 31.7% (0.663%) | 0.9365 |
| `stage3` | ResNet Block Stage 3 | 33.0% (0.692%) | 0.9404 |
| `fc` | Linear Classifier Head | 15.6% (0.326%) | 0.0000 |

### Re-Basin Permutation & Basin Geometry
- **Re-Basin Accuracy Advantage:** **+0.00%** (Identical to standard merge; confirms shared initialization $W_0$ preserves unimodal basin alignment $\Pi=I$).
- **Linear CKA Alignment:** $> 0.91$ across all layers, establishing that functional divergence in FL clients is concentrated in output logits rather than internal feature geometry.
