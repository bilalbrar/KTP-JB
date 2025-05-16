# BCU Data Analysis Systems

A web application providing access to machine learning models for Energy Performance Certificate (EPC) prediction and drug review sentiment analysis, secured with authentication.

## Features

- **EPC Rating Prediction**: Analyze and predict potential energy ratings for properties in Birmingham
- **Drug Review Sentiment Analysis**: Classify drug reviews as positive, neutral, or negative
- **Secure Authentication**: JWT-based login system to protect the dashboards and APIs
- **Responsive Web Interface**: User-friendly dashboards for both analysis systems
- **API Access**: Endpoints for programmatic access to the prediction models

## Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bcu-analysis.git
   cd bcu-analysis
   ```

2. Build and run with Docker Compose:
   ```
   docker-compose up --build
   ```

3. Access the application at http://localhost:5000

### Manual Setup (without Docker)

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

## Usage

1. Access the web interface at http://localhost:5000
2. Login with the credentials:
   - Username: `admin`
   - Password: `password`
3. Select the dashboard you want to use:
   - EPC Dashboard: Search properties and view predicted energy ratings
   - Drug Review Dashboard: Analyze sentiment in drug reviews

## API Documentation

### Authentication

All API endpoints require a JWT token, obtained by logging in:

```
POST /login
Content-Type: application/json

{
  "username": "admin",
  "password": "password"
}
```

Response:
```json
{
  "token": "your-jwt-token"
}
```

Use this token in subsequent requests:
```
GET /api/epc/properties?query=birmingham
X-Access-Token: your-jwt-token
```

### EPC Endpoints

- `GET /api/epc/properties?query=<search_term>`: Search for properties
- `POST /api/epc/predict`: Predict potential energy rating for a property

### Drug Review Endpoints

- `POST /api/drug/predict`: Analyze sentiment in a drug review

## Project Structure

- `/templates`: HTML templates for the web interface
- `/models`: Saved machine learning models
- `/Birmingham Energy Performance`: Dataset for EPC prediction
- `/Drug Reviews Dataset`: Dataset for sentiment analysis
- `app.py`: Main Flask application
- `epc.py`: EPC model training script
- `drug.py`: Drug sentiment model training script

## Technologies Used

- **Backend**: Flask, JWT
- **Machine Learning**: scikit-learn, NLTK
- **Data Processing**: pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Containerization**: Docker

## License

MIT License

## Acknowledgments

- Birmingham City University
- Jhoots Pharmacy

