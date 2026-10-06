# CrystalCall

Code and traces for the paper **Where Should a Diffusion-LM Agent Start Its Tools?**
Marek Šuppa and Jaroslav Kopcan.
NeurIPS 2026 Workshop on SLMs for Agentic Systems (SLM-Agents), Paris.

Paper: https://openreview.net/forum?id=nVaHns2ZDR

A diffusion language model holds a draft of its whole answer at every decoding
step, so a tool call can be readable before generation ends. CrystalCall
launches side-effect-free tool calls early and hands a result to the agent only
if the final program makes exactly that call.

**Status:** the code and recorded traces will be published in this repository
soon.

## Citation

```bibtex
@inproceedings{suppa2026crystalcall,
  author    = {Marek {\v{S}}uppa and Jaroslav Kopcan},
  title     = {Where Should a Diffusion-{LM} Agent Start Its Tools?},
  booktitle = {NeurIPS 2026 Workshop on SLMs for Agentic Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=nVaHns2ZDR}
}
```
