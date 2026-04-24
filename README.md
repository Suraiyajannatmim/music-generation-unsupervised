# Unsupervised Neural Network for Multi-Genre Music Generation

Course project for CSE425/EEE474 Neural Networks.

## Contents
- `notebooks/music_generation_full_4_tasks_colab.ipynb` - complete Colab workflow.
- `src/` - modular source code for preprocessing, models, training, evaluation, and generation.
- `outputs/generated_midis/` - generated MIDI samples.
- `outputs/plots/` - result plots.
- `outputs/tables/` - metrics and final comparison CSV files.
- `outputs/survey_results/` - listening survey sheet.
- `report/final_report.pdf` - final project report.

## Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/music_generation_full_4_tasks_colab.ipynb
```
Recommended: run in Google Colab with GPU and put public MIDI files in genre folders.

## GitHub upload
```bash
git init
git add .
git commit -m "Initial submission"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/music-generation-unsupervised.git
git push -u origin main
```
