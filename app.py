from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
import pandas as pd
import joblib, os, jwt, secrets, re
from datetime import datetime, timedelta
from functools import wraps
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)

@app.route('/public/<path:filename>')
def serve_public(filename):
    return send_from_directory('public', filename)

# Setup paths for all models
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# EPC model paths
epc_model_path = os.path.join(models_dir, 'epc_best_model.pkl')
epc_le_path = os.path.join(models_dir, 'epc_label_encoder.pkl')
epc_feature_path = os.path.join(models_dir, 'epc_feature_metadata.pkl')

# Sentiment analysis model path
sentiment_model_path = os.path.join(models_dir, 'best_sentiment_model.pkl')

# Initialize models
epc_model = None
epc_le = None
epc_feature_metadata = None
sentiment_model = None

# Text processing for sentiment analysis
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^a-z\s]', ' ', text.lower())
    return ' '.join(lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words)

# Load models - with error handling for each model separately
try:
    epc_model = joblib.load(epc_model_path)
    epc_le = joblib.load(epc_le_path)
    epc_feature_metadata = joblib.load(epc_feature_path)
    print("EPC model and components loaded successfully")
except FileNotFoundError as e:
    print(f"Warning: EPC model components not found: {e}")
    print("EPC prediction functionality will be limited.")

try:
    sentiment_model = joblib.load(sentiment_model_path)
    print("Sentiment analysis model loaded successfully")
except FileNotFoundError as e:
    print(f"Warning: Sentiment model not found: {e}")
    print("Sentiment analysis functionality will be limited.")

# Mock database for demonstration purposes
# In a real application, you would use a proper database
# This includes a mock property database and user credentials
MOCK_USERS = {
    'admin': {
        'password': 'password',
        'role': 'admin'
    }
}

# Sample property database (would come from your actual data)
MOCK_PROPERTIES = {}
try:
    # Load a small subset of properties from your dataset for demo purposes
    df_path = r"C:\Users\bilal\BCU\Birmingham Energy Performance\certificates.csv"
    df = pd.read_csv(df_path, low_memory=False, nrows=1000)  # Limit to 1000 rows for demo
    
    for _, row in df.iterrows():
        lmk = row.get('LMK_KEY')
        if pd.notna(lmk):
            # prefer ADDRESS1, fallback to ADDRESS
            addr = row.get('ADDRESS1') if pd.notna(row.get('ADDRESS1')) else row.get('ADDRESS')
            pc   = row.get('POSTCODE')
            # only include if both address and postcode exist
            if pd.notna(addr) and pd.notna(pc):
                MOCK_PROPERTIES[lmk] = {
                    'address': addr,
                    'postcode': pc,
                    'current_rating': row.get('CURRENT_ENERGY_RATING', 'Unknown')
                }
    print(f"Loaded {len(MOCK_PROPERTIES)} properties for demo")
except Exception as e:
    print(f"Warning: Could not load property data: {e}")
    # Provide some mock properties if the file loading fails
    MOCK_PROPERTIES = {
        'SAMPLE1234': {
            'address': '123 Sample Street',
            'postcode': 'B12 3DE',
            'current_rating': 'D'
        },
        'SAMPLE5678': {
            'address': '456 Example Road',
            'postcode': 'B45 6FG',
            'current_rating': 'C'
        }
    }

# Token required decorator for protected routes
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token is in headers
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        
        # Also check cookies for web interface
        if not token and request.cookies.get('token'):
            token = request.cookies.get('token')
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            # Decode the token
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

# Main routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # For API requests
        if not username and request.headers.get('Content-Type') == 'application/json':
            auth = request.get_json()
            username = auth.get('username')
            password = auth.get('password')
        
        # Check if the user exists and password is correct
        if username in MOCK_USERS and MOCK_USERS[username]['password'] == password:
            # Generate a token
            token = jwt.encode({
                'user': username,
                'exp': datetime.utcnow() + timedelta(hours=24)
            }, app.config['SECRET_KEY'], algorithm="HS256")
            
            # For API use, return the token
            if request.headers.get('Content-Type') == 'application/json':
                return jsonify({'token': token})
            
            # For web form, set a cookie and redirect to dashboard selection
            resp = redirect(url_for('dashboard_selection'))
            resp.set_cookie('token', token)
            return resp
        
        # Invalid credentials
        if request.headers.get('Content-Type') == 'application/json':
            return jsonify({'message': 'Invalid credentials'}), 401
        
        return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

# Dashboard selection page
@app.route('/select-dashboard')
@token_required
def dashboard_selection():
    return render_template('dashboard_selection.html')

# EPC Dashboard
@app.route('/epc-dashboard')
@token_required
def epc_dashboard():
    return render_template('epc_dashboard.html', properties=MOCK_PROPERTIES)

# Drug sentiment dashboard
@app.route('/drug-dashboard')
@token_required
def drug_dashboard():
    return render_template('drug_dashboard.html')

# API Routes
# EPC API
@app.route('/api/epc/predict', methods=['POST'])
@token_required
def epc_predict():
    data = request.get_json() or {}
    lmk = data.get('LMK_KEY')
    if not lmk or lmk not in MOCK_PROPERTIES:
        return jsonify({'error': 'Invalid LMK_KEY'}), 400

    props = MOCK_PROPERTIES[lmk]
    rating_map = {'A':7,'B':6,'C':5,'D':4,'E':3,'F':2,'G':1}
    score = rating_map.get(props['current_rating'], 4)

    fn = epc_feature_metadata['numeric_features']
    cf = epc_feature_metadata['categorical_features']
    num_feats = {f:[0.0] for f in fn}
    for k,v in [('TOTAL_FLOOR_AREA',100), ('ENERGY_CONSUMPTION_CURRENT',250), ('num_recommendations',3)]:
        if k in num_feats: num_feats[k] = [v]
    cat_feats = {f:['Missing'] for f in cf}
    cat_feats['CURRENT_ENERGY_RATING'] = [props['current_rating']]

    sample = pd.DataFrame({**num_feats, **cat_feats})
    idx    = epc_model.predict(sample)[0]
    pred   = epc_le.inverse_transform([idx])[0]

    return jsonify({
        'lmk_key': lmk,
        'current_rating': props['current_rating'],
        'predicted_potential_rating': pred
    })

@app.route('/api/epc/properties', methods=['GET'])
@token_required
def get_epc_properties():
    # Simple API to search properties
    query = request.args.get('query', '').lower()
    
    if not query:
        return jsonify({'message': 'Please provide a search query'}), 400
    
    results = []
    for lmk_key, property in MOCK_PROPERTIES.items():
        if (query in lmk_key.lower() or 
            query in property['address'].lower() or 
            query in property['postcode'].lower()):
            results.append({
                'lmk_key': lmk_key,
                'address': property['address'],
                'postcode': property['postcode'],
                'current_rating': property['current_rating']
            })
    
    return jsonify({'properties': results[:20]})  # Limit to 20 results

# Drug sentiment API
@app.route('/api/drug/predict', methods=['POST'])
@token_required
def drug_predict():
    try:
        data = request.get_json()
        
        if not data or 'review' not in data:
            return jsonify({'error': 'No review text provided'}), 400
        
        review_text = data['review']
        
        # Clean the text
        cleaned_review = clean_text(review_text)
        
        # Check if model is loaded
        if sentiment_model is None:
            return jsonify({
                'error': 'Sentiment model not loaded',
                'message': 'For demo purposes, returning a mock prediction'
            }), 500
        
        # Use the model to predict
        try:
            prediction = sentiment_model.predict([cleaned_review])[0]
        except Exception as e:
            # Fallback to mock prediction
            import random
            sentiments = ['positive', 'neutral', 'negative']
            prediction = random.choice(sentiments)
            return jsonify({
                'warning': f'Error using model: {str(e)}',
                'sentiment': prediction,
                'note': 'This is a mock prediction due to model error'
            })
        
        return jsonify({
            'sentiment': prediction,
            'review': review_text[:100] + ('...' if len(review_text) > 100 else '')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
