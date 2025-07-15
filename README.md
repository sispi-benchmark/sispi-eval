# SISPI Evaluation Code

This repository provides the evaluation scripts for the paper:  
**"Measuring Text-Image Retrieval Fairness with Synthetic Data"**  
*Lluis Gomez – Computer Vision Center, Universitat Autonoma de Barcelona*

📄 DOI: [https://doi.org/10.1145/3726302.3730030](https://doi.org/10.1145/3726302.3730030)  
🌐 Project: [https://sispi-benchmark.github.io/sispi-benchmark/](https://sispi-benchmark.github.io/sispi-benchmark/)  
📦 Dataset: [https://huggingface.co/datasets/lluisgomez/SISPI](https://huggingface.co/datasets/lluisgomez/SISPI)

---

## 🧪 Setup

Create a conda environment with all required dependencies:

```bash
conda env create -f environment.yml
conda activate sispi-eval
```

---

## ▶️ Run Evaluation

To evaluate using a pretrained CLIP model:

```bash
python eval_clip_demo.py
```

To evaluate a fine-tuned model:

```bash
python eval_clip_demo.py --train_output_dir path/to/checkpoint_dir
```

Results will be printed and saved (if output directory is provided).

---

## 📜 Citation

```bibtex
@inproceedings{10.1145/3726302.3730030,
  author = {Gomez, Lluis},
  title = {Measuring Text-Image Retrieval Fairness with Synthetic Data},
  year = {2025},
  url = {https://doi.org/10.1145/3726302.3730030},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series = {SIGIR '25'}
}
```
