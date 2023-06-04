# DocMAE

Unofficial implementation of **DocMAE: Document Image Rectification via Self-supervised Representation Learning**

https://arxiv.org/abs/2304.10341

## TODO

- [x] Document background segmentation network using U2 net
- [ ] Synthetic data generation for self-supervised pre-training (in progress: 1M done)
- [ ] Pre-training (training)
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

Run training via:
`python pretrain.py --output_dir ./out_dir/ --train_dir ./train_masked --validation_dir ./val_masked --remove_unused_columns False --do_train --do_eval --base_learning_rate 1.5e-4 --lr_scheduler_type cosine --weight_decay 0.05 --num_train_epochs 800 --warmup_ratio 0.05 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --logging_strategy steps --logging_steps 10 --evaluation_strategy epoch --save_strategy epoch --load_best_model_at_end True --save_total_limit 3 --seed 1337 --config_overrides="image_size=288" --max_eval_samples=10000`

Visualize trained model using https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb

# Acknowledgement

Test documents come from DIR300 dataset https://github.com/fh2019ustc/DocGeoNet