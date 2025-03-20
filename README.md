# Named Entity Recognition (NER) System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-supported-blue.svg)](https://www.docker.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-model-yellow.svg)](https://huggingface.co/)

A production-ready Named Entity Recognition system that identifies entities like persons, organizations, locations, and more from text data.

## Model Information

This project uses a pre-trained NER model hosted on Hugging Face. There is no local model file included in this repository - the system automatically downloads the model from Hugging Face during initialization.

- Model: BERT-based token classification fine-tuned for NER
- Performance: 92% F1 score on CoNLL-2003 evaluation set
- Hosted at: `huggingface.co/my-username/ner-bert-model`

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- Docker (optional, for containerized deployment)

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/ner-system.git
cd ner-system
```

### Step 2: Set up virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```


## Running the Application

### Option 1: Run with Python
```bash
# Start the API server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Access the API documentation
# Open http://localhost:8000/docs in your browser
```


## Usage Examples

### Using the API with curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple is looking at buying U.K. startup for $1 billion"}'
```

### Using the API with Python
```python
import requests

# Make prediction with the API
response = requests.post(
    "http://localhost:8000/predict",
    headers={"Authorization": "Bearer your_api_key"},
    json={"text": "Apple is looking at buying U.K. startup for $1 billion"}
)

print(response.json())
# Output: {"entities": [{"text": "Apple", "start": 0, "end": 5, "label": "ORG"}, ...]}
```

## Project Structure

```
NER_MODEL_EXTRA/
├── docs/                   # Documentation files
│   └── Documentation.md    # Detailed documentation
├── src/                    # Source code
│   ├── api/                # API implementation
│   │   └── main.py         # FastAPI application
│   └── frontend/           # Frontend implementation
│       └── app.py          # Streamlit interface
|
workflow
|   └── wokflow.png          # flowchart image
|   └── ner_workflow.md      # full workflow
links
|  └── links              #links to deployed website
|
├── logs/                   # System logs
│   └── api_*_*.log         # Timestamped log files
├── SmallModelWithoutHyperParameterTuning (1).ipynb  # Small model training notebook
├── MainModelWithTrainingAndHyperparameterTuning (1).ipynb  # Main model training notebook
├── run_inference.py        # Inference implementation 
├── README.md               # Project README
└── requirements.txt        # Dependencies
```

## Troubleshooting

- **Model download issues**: Check your internet connection and Hugging Face token
- **Memory errors**: The model requires at least 2GB of RAM
- **Slow first request**: The first request may be slow as the model is loaded into memory
- **Render deployment issues**: When using Render's free plan, the API may experience slow loading times after periods of inactivity (cold starts). Some requests may fail with "memory limit exceeded" errors due to the limited RAM allocation in the free tier.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
