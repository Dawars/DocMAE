# DocMAE

Unofficial implementation of **DocMAE: Document Image Rectification via Self-supervised Representation Learning**

https://arxiv.org/abs/2304.10341

## TODO

- [x] Document background segmentation network using U2 net
- [ ] Synthetic data generation for self-supervised pre-training (in progress: 900k done)
- [ ] Pre-training
- [ ] Fine-tuning for document rectification
- [ ] Evaluation
- [ ] Code clean up and documentation
- [ ] Model release

## Demo

Find a jupyter notebook at [demo/background_segmentation.ipynb](demo/background_segmentation.ipynb)

## Data

### Pre-training

- 3411482 pages from ~1M documents from Docile dataset (https://github.com/rossumai/docile)
- Rendered with Doc3D https://github.com/Dawars/doc3D-renderer
- 558 HDR env lighting from https://hdri-haven.com/

# Acknowledgement

Test documents come from DIR300 dataset https://github.com/fh2019ustc/DocGeoNet