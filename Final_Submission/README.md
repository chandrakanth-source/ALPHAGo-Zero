# Final Submission — AlphaGo Zero (CSE 311 Artificial Intelligence)

Research paper review and implementation of *Mastering the game of Go without human knowledge*
(Silver et al., Nature 2017) on a 12×12 board. Indian Institute of Information Technology Kottayam;
course instructor Dr. A V Prajeesh.

**Team:** Shaik Shafreed (2024BCS0113) · Chandrakanth (2024BCS0125) · Arya Vardhan (2024BCS0129) ·
Shaik Humair Fatah (2024BCS0105) · Lakavath Harshithan (2024BCS0321)

## Files

| File | Contents |
|---|---|
| `AlphaGoZero_Final_Report.docx` / `.pdf` | Final report on the faculty template (21 pages) |
| `AlphaGoZero_Presentation.pptx` | Presentation (20 slides); each slide's script is in its speaker notes |
| `AlphaGoZero_Presentation_Scripts.docx` | Speaking script for each team member, with slide allocation and timings |
| `build/` | Scripts that regenerate the report, figures and scripts (see below) |

## Trained models

All models are in [`../models/`](../models) and stored with Git LFS (run `git lfs pull` after cloning).

| File | Description | Evaluation result |
|---|---|---|
| `pre_curriculum_best.pt` | Best model before the curriculum; opponent for iterations 3–4 | — |
| `model_iteration_3.pt` | Stage 2 candidate | 30%, rejected |
| `model_iteration_4.pt` | Stage 2 candidate | 70%, **promoted** |
| `model_iteration_5.pt` | Stage 3 candidate | 45%, rejected |
| `model_iteration_6.pt` | Stage 3 candidate | 50%, rejected |
| `model_iteration_7.pt` | Stage 4 candidate | 55%, **promoted** |
| `model_iteration_8.pt` | Stage 4 candidate | 52.5%, rejected |
| `model_iteration_9.pt` | Stage 5 candidate | 72.5%, **promoted** — final best |
| `model_iteration_10.pt` | Stage 5 candidate | 50%, rejected |
| `latest_model.pt` | Current best model (identical to iteration 9) | — |

Iterations 0–2 and the files numbered 11–100 come from an earlier pipeline that predates the bug fixes
described in the report, and are not part of the reported results.

## Rebuilding the documents

```bash
cd Final_Submission/build
python -m venv .venv && .venv/bin/pip install matplotlib python-docx defusedxml lxml
.venv/bin/python make_figures.py figures
.venv/bin/python build_report.py "<path to Course Project Report Template.docx>" figures ../AlphaGoZero_Final_Report.docx
```

Report text is in `report_content.py`; team members and presentation scripts are in `scripts_content.py`.
The report build needs LibreOffice and `pdftotext` to compute the contents-page numbers.
