# DocMAE
Unofficial implementation of **DocMAE: Document Image Rectification via Self-supervised Representation Learning**

https://arxiv.org/abs/2304.10341

## TODO
- [ ] Document background segmentation network using U2 net (almost done)
- [ ] Synthetic data generation for self-supervised pre-training (in progress)
- [ ] Pre-training
- [ ] Fine-tuning for document rectification
- [ ] Evaluation
- [ ] Code clean up and documentation
- [ ] Model release


## Data
### Pre-training
- 3411482 pages from ~1M documents from Docile dataset (https://github.com/rossumai/docile)
- Rendered with Doc3D https://github.com/Dawars/doc3D-renderer
- 558 HDR env lighting from https://hdri-haven.com/
