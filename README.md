This project is designed for Langerhans cell activation level grading, and its complete execution process is outlined below.

The dataset resource is located at https://zenodo.org/records/17861993.

**Step 1: Train segmentation task (Segmentation_model) using CORN‑Pro, and test on IID‑Seg and OOD‑Seg**  
- Set `config.py` (specify `--dataset`, `--env`).  
- Run `python main.py` (two consecutive training phases).  
- Use `predict_metric.py` to evaluate generalization performance.

**Step 2: Train vision‑based grading task (Vision_grading_model) using CORN‑LCs, and test on DED, DM, HSK, PCS**
- Run `python train.py` with `'checkpoint': False` for initial training (specify `--dataset`).  
- Use `Predict_metric.py` to evaluate generalization performance.  
- Run `python utils.interpretability_CAM.py` to obtain level‑based Grad‑CAM maps.  
- Run `python cnn_visualization.gradcam_IMAGE.py` to obtain ROI images (required for fine‑tuning) and CAM images.  
- Run `python train.py` with `'checkpoint': True` for fine‑tuning (only set `--checkpoint`).

**Step 3: Train tokenizer (Generative_model/Train_tokenizer) to understand numerical quantitative indicators**  
- Run `python Generative_model.Train_tokenizer.generate_train_data.py` to prepare training data.  
- Run `python train.py` to obtain the well‑trained numerical tokenizer.

**Step 4: Train text‑based grading task (Generative_model) using corpus**
- Use `Segmentation_model.inference_or_label.py` (`inference_or_label_IMAGE.py`) to obtain segmentation maps for all datasets in `Dataset.Segmentation_task`.  
- Use `Vision_grading_model.inference_or_label.py` (`inference_or_label_IMAGE.py`) to obtain grading labels for all datasets in `Dataset.Grading_task.Activation.Grading_dataset`.  
- Follow the six steps in `Dataset_generation_Vision_Text` to create the generative model corpus (two stages) and the paired image‑text multimodal dataset.  
- Set `config.config_LCs_grading.json`.  
- Run `train Stage1_generative_pretraining.py` for generative pre‑training (only set `config_file`, then run).  
- *(Optional)* Run `Eval_generative_pretraining.py` to evaluate the generative model’s performance.  
- Run `train Stage2_finetune.py` to fine‑tune the generative model in a discriminative classification manner.  
- *(Optional)* Run `Eval_finetuning.py` to evaluate the discriminative classification performance.

**Step 5: Train multimodal fusion task (Multimodal_fusion) using corpus**
- Run `python Logistic_regression.py` to obtain the final grading results.



