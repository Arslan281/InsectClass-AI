

\# InsectClass AI



AI-powered insect pest identification using feedforward artificial neural networks.



\## Project Overview



InsectClass AI is a university Artificial Neural Networks project that identifies insects from 15 selected IP102 pest categories.



It compares a Single-Layer ANN and a Multilayer ANN through a React website and a Python FastAPI backend. An additional insect/non-insect detection model helps reject unrelated images.



\## Technologies



\- Python 3.12 and TensorFlow/Keras

\- NumPy and other dependencies in `requirements.txt`

\- FastAPI and Uvicorn

\- React, Vite, and JavaScript

\- IP102 insect pest dataset



\## Final Test Results



Both classifiers were evaluated on 7,841 held-out insect images.



| Model | Accuracy | Macro F1 |

|---|---:|---:|

| Improved Single-Layer ANN | 31.44% | 26.45% |

| Improved Multilayer ANN | 43.54% | 39.86% |



These are experimental classification results, not production-level reliability guarantees.



\## Features



\- Insect Explorer with predictions from both feedforward ANN models

\- Insect/non-insect detection gate

\- Insect Library with example images

\- Developer Studio with evaluation charts and confusion matrices



\## Running Locally



Backend (from the repository root):



```powershell

py -3.12 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

python -m uvicorn backend.main:app --reload

```



Frontend (in a second terminal):



```powershell

cd frontend

npm install

npm run dev

```



Open the local Vite address shown in the terminal, normally http://localhost:5173.



\## Dataset and Reproducibility



The project uses 15 selected classes from IP102. Large raw datasets and generated NumPy arrays are excluded from GitHub.



Final inference models and training/evaluation scripts are included. Some experiment paths may require adjustment for another computer.



\## Future Development



InsectClass AI is currently an academic research prototype. Future plans include improving field-image accuracy, expanding supported species, and developing the platform into a commercially viable identification service.



