# Lightweight Bias Mitigation using GAN / Augmentation (AIM-Fair Modification)

##  Overview

This project is a modified implementation of the AIM-Fair framework for bias mitigation in deep learning models.

The original AIM-Fair method uses diffusion-based synthetic data generation, which requires high computational resources (e.g., A100 GPUs).
In this project, we replace the diffusion-based approach with a lightweight alternative using data augmentation and simple synthetic strategies.

---

##  Key Contributions

* Replaced diffusion models with lightweight data augmentation techniques
* Adapted the pipeline to the UTKFace dataset
* Reduced GPU requirements (runs on Colab T4 GPU)
* Maintained comparable accuracy with reduced computational cost

---

##  Project Structure

```
AIM-Fair-Lightweight/
│
├── dataloader/                # Data loading modules
├── models/                   # Model architectures
│
├── main.py                   # Main training script
├── load_data_UTKFace.py      # UTKFace dataset loader
├── Fairness_Metrics.py       # Fairness evaluation metrics
├── Generate_Model.py         # Model generation utilities
│
├── gan_generate.py           # GAN-based synthetic data (optional)
├── synthetic.py              # Synthetic data utilities
│
├── runner.sh                 # Script to run experiments
├── README.md                 # Project documentation
├── LICENSE
```

---

##  Installation

```bash
pip install torch torchvision matplotlib numpy pillow
```

---

##  How to Run

```bash
python main.py \
--real-data-path /path/to/UTKFace \
--synthetic-data-root /path/to/synthetic \
--seeds 1 \
--milestones 5 10 \
--job-id test_run
```

---

##  Results

### Baseline (AIM-Fair on UTKFace)

* Overall Accuracy: **94.70**
* Worst-group Accuracy: **93.81**
* Equalized Odds: **1.16**
* STD: **0.65**

### Proposed Method (Lightweight / Augmentation)

* Overall Accuracy: **94.55**
* Worst-group Accuracy: **92.88**
* Equalized Odds: **1.52**
* STD: **1.17**

---

##  Analysis

The proposed method achieves performance comparable to the original AIM-Fair model while significantly reducing computational requirements.

* Slight drop in fairness metrics
* Similar overall accuracy
* Much more efficient and reproducible

---

##  Conclusion

This work demonstrates that bias mitigation can be achieved without relying on heavy diffusion models.
Lightweight approaches provide a practical alternative for real-world deployment.

---

##  Notes

* Dataset (UTKFace) is not included in this repository
* Synthetic data folders are optional
* Ensure correct dataset paths before running

---

##  License

This project follows the same license as the original AIM-Fair repository.

---

##  Acknowledgement

Based on the original AIM-Fair framework, with modifications for efficiency and accessibility.
