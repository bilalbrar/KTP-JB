# BCU Data Analysis Systems

A Flask web application that provides two secure dashboards:

1. **EPC Prediction**: Predict potential Energy Performance Certificate ratings for properties in Birmingham using trained ML models.  
2. **Drug Review Sentiment Analysis**: Classify user-entered drug reviews as positive, neutral, or negative.

## Features
- JWT-based authentication (login logic in `app.py`).  
- Interactive EPC dashboard with search by `LMK_KEY`, address, or postcode.  
- Drug review input with instant sentiment results.  
- REST API endpoints for programmatic access.  
- Model training scripts: `epc.py` and `drug.py`.  
- Frontend templates under `templates/`, static assets in `public/`.

## Prerequisites
- Python 3.9+  
- Virtual environment tool (venv or conda)  
- Data files placed in (default paths):
  - `Birmingham Energy Performance/certificates.csv`
  - `Birmingham Energy Performance/recommendations.csv`
  - `Drug Reviews Dataset/drugsComTrain_raw.csv`

## Installation & Setup
```bash
# 1. Clone the repo
git clone https://github.com/bilalbrar/KTP-JB.git
cd KTP-JB

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare your datasets in the folders:
#    BCU\Birmingham Energy Performance\ and BCU\Drug Reviews Dataset\

# 5. (Optional) Train or retrain models
python epc.py     # outputs models/*.pkl
python drug.py    # outputs models/best_sentiment_model.pkl

# 6. Run the Flask application
python app.py
```

## Project Structure
```
BCU/
├── models/                     # Saved model artifacts
│   ├── epc_best_model.pkl
│   ├── epc_preprocessor.pkl
│   ├── epc_label_encoder.pkl
│   ├── epc_feature_metadata.pkl
│   └── best_sentiment_model.pkl
├── public/                     # Static assets (logos, CSS, JS)
├── templates/                  # Jinja2 HTML templates
│   ├── index.html
│   ├── login.html
│   ├── dashboard_selection.html
│   ├── epc_dashboard.html
│   └── drug_dashboard.html
├── epc.py                      # EPC data preprocessing & model training
├── drug.py                     # Drug review preprocessing & model training
├── app.py                      # Flask app serving dashboards & APIs
├── requirements.txt            # Python dependencies
└── README.md

```

## Usage

### Web Interface
1. Navigate to `http://localhost:5000`  
2. Login with default credentials:
   ```
   username: admin
   password: password
   ```  
3. Choose a dashboard:
   - **EPC Dashboard**: Search properties & view predictions  
   - **Drug Review Dashboard**: Enter review text for sentiment analysis  

### API Endpoints
All endpoints require a JWT in header `x-access-token` (obtained via `/login`).

#### Authentication
```
POST /login
Content-Type: application/json

{ "username":"admin", "password":"password" }
```
Response:
```json
{ "token": "<JWT_TOKEN>" }
```

#### EPC APIs
- `GET /api/epc/properties?query=<term>`  
  → List up to 20 matching properties  
- `POST /api/epc/predict`  
  Body:
  ```json
  { "LMK_KEY": "<property_key>" }
  ```

#### Drug Review API
- `POST /api/drug/predict`  
  Body:
  ```json
  { "review": "Your drug review text here" }
  ```

## Notes
- Model training scripts write to `models/`. Re-run them if data or parameters change.  
- Adjust dataset paths in `epc.py` and `app.py` if your folder layout differs.  
- Extend authentication logic in `app.py` for production use.

## License
MIT License

