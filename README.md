# The Manga Task \[Advanced\] --- My Learning Journey

## About the project

This project is my attempt at solving the Manga Task \[Advanced\]. The
goal is to process three consecutive English manga pages, extract the
text in reading order, and keep speaker labels consistent across the
pages.

I wanted to build something that actually works, but the process was not
as straightforward as I first expected. A big part of this project was
figuring out what I could realistically run on my own laptop and
learning the tools needed to make it happen.

## How I approached it

I first tried building the solution with **n8n**. It was a useful
starting point, but I ran into complications when I started thinking
about the model and inference requirements. Some of the approaches I
explored were better suited to machines with a dedicated GPU, and I do
not have an NVIDIA GPU available on my laptop.

Instead of stopping there, I started looking for a way to make the
project work with the hardware I already had. That meant learning more
about **Python**, understanding how the different parts of an ML
pipeline fit together, and exploring smaller, more practical models and
CPU-friendly tools.

I had to learn as I went. I spent time understanding how data is loaded,
how OCR can extract text from images, how reading order can be
reconstructed, and how a model can help distinguish narration from
dialogue. I also learned that generating output is only one part of the
task: the output has to follow the required format, and the results need
to be checked against the evaluation script.

## What I built

The current project is organised into separate parts so that I can test
and improve them individually:

-   **Dataset inspection and validation** to check the supplied files
    and images.
-   **Image preprocessing and OCR** to help extract text from manga
    pages.
-   **Reading-order logic** to arrange detected text in the intended
    order.
-   **Speaker-tracking logic** to keep labels consistent within a
    three-page sequence.
-   **A narration classifier** as the learned component in the pipeline.
-   **Evaluation scripts** to compare the baseline with the adapted
    version.
-   **Submission validation** to check the generated JSONL file before
    submission.

I also kept the project scripts, configuration, outputs, model artifact,
and logs in separate folders. This makes it easier to understand what
each part does and to debug problems without treating the whole project
as one large block of code.

## The main challenge: working without a dedicated GPU

One of the biggest lessons for me was that I could not simply choose a
large model and expect it to run well on my laptop. My machine uses
Intel UHD 620 integrated graphics, so I had to pay attention to memory
use and CPU compatibility instead of assuming that CUDA or a powerful
GPU was available.

This changed the way I approached the problem. I focused on a lighter
pipeline and on understanding each stage rather than relying on a large
model to do everything. I learned that choosing a model is not just
about how capable it is; it also has to fit the hardware, the task
rules, and the time available.

## System Architecture & Pipeline Overview

The project is structured into modular, reusable components:

```
Manga-task/
├── config/
│   └── config.json                  # Pipeline configuration & hyperparameters
├── development/
│   ├── images/                      # 80 development sequences (240 pages)
│   └── labels.jsonl                 # Ground truth annotations
├── models/
│   └── narration_classifier.joblib  # Trained learned component weights
├── outputs/
│   ├── baseline_dev_predictions.jsonl
│   ├── adapted_dev_predictions.jsonl
│   └── test_predictions.jsonl       # Final predictions on 15 test sequences
├── logs/
│   └── train_metrics.json           # Training logs and validation performance
├── scripts/
│   ├── inspect_dataset.py           # Dataset statistics & speaker breakdown
│   ├── validate_dataset.py          # Data integrity & image verification
│   ├── run_baseline.py              # Baseline pipeline execution & scoring
│   ├── train.py                     # Training of the learned narration classifier
│   ├── evaluate.py                  # Official comparative evaluation (score.py)
│   ├── predict.py                   # Test sequence inference & submission generation
│   └── validate_submission.py       # Strict schema & rule compliance validator
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Leakage-free sequence loader & splitters
│   ├── image_preprocessing.py       # Contrast enhancement & bubble detection
│   ├── ocr_engine.py                # CPU-optimized EasyOCR extraction
│   ├── reading_order.py             # Manga Right-to-Left / Top-to-Bottom sorter
│   ├── speaker_tracking.py          # Sequence-wide character consistency tracker
│   ├── model.py                     # Narration & Dialogue Classifier (Learned ML)
│   └── submission_writer.py         # JSONL formatting & validator
├── tests/
│   └── test_pipeline.py             # Automated unit and integration tests
├── requirements.txt                 # Dependencies
├── sequences.json                   # Dataset split index
├── sample_submission.jsonl          # Submission template
├── score.py                         # Official evaluation script
└── README.md
```

---

## What I learned

This project helped me learn several things beyond just writing code:

-   How to break a larger ML problem into smaller modules.
-   How to work with Python scripts, dependencies, configuration files,
    and project folders.
-   The role of OCR in extracting text from images.
-   Why reading order and speaker attribution are separate problems.
-   How a learned classifier differs from hand-written rules.
-   Why training and validation data should be split by complete
    sequence to avoid leakage.
-   How to compare a baseline with an adapted model using the provided
    evaluation script.
-   Why a valid output file does not automatically mean that the
    predictions are accurate.
-   How to investigate limitations instead of assuming that a tool or
    model will work on my machine.

I also became more comfortable reading errors, checking files, and
improving the project step by step. I am still learning, but this
process gave me a much clearer idea of how a practical ML workflow is
put together.

## Results and outputs

The project currently includes:

-   `outputs/baseline_dev_predictions.jsonl` --- baseline predictions on
    development data.
-   `outputs/adapted_dev_predictions.jsonl` --- predictions from the
    adapted pipeline.
-   `outputs/test_predictions.jsonl` --- generated predictions for the
    test sequences.
-   `models/narration_classifier.joblib` --- saved narration-classifier
    artifact.
-   `logs/train_metrics.json` --- recorded training metrics.
-   `score.py` --- the supplied evaluation script.

The existing project report records a held-out narration-classifier
macro F1 of **0.7879**. It also records `balanced_joint_f1` improving
from **0.3719** for the baseline to **0.4014** for the adapted pipeline.
These figures should be treated as the project's recorded results and
can be reproduced by running the relevant scripts.

I still see this as a work in progress. Text extraction, reading order,
and especially consistent speaker identification can be difficult in
manga pages, so there is room to test more examples and improve the
pipeline.

## How to run it

Use Windows PowerShell from the project root.

### 1. Install dependencies

``` powershell
pip install -r requirements.txt
```

### 2. Validate the dataset

``` powershell
python scripts/validate_dataset.py
```

### 3. Inspect the dataset

``` powershell
python scripts/inspect_dataset.py
```

### 4. Run the tests

``` powershell
python tests/test_pipeline.py
```

### 5. Run the baseline

``` powershell
python scripts/run_baseline.py
```

### 6. Train the learned component

``` powershell
python scripts/train.py
```

### 7. Evaluate the pipeline

``` powershell
python scripts/evaluate.py
```

### 8. Generate test predictions

``` powershell
python scripts/predict.py
```

### 9. Validate the final output

``` powershell
python scripts/validate_submission.py --predictions outputs/test_predictions.jsonl
```

Run these commands in order and check the terminal output. If a command
fails, resolve that issue before assuming the next stage has completed
successfully.

## Final note

I started this project by trying a tool that I already knew about, ran
into limitations, and then had to learn new things to keep moving. I
learned Python, explored different model options, worked through CPU
constraints, and tried to understand the pipeline rather than treating
it as a black box.

As a first-year student, I know I still have a lot to learn. But this
project gave me practical experience with problem-solving, debugging,
model evaluation, and making technical choices based on real
constraints. The most valuable part for me was not just getting files
generated; it was understanding more of what it takes to build and test
an ML solution.
