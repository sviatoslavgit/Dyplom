"""
Fraud Detection Server - PostgreSQL Integration
Connects to PostgreSQL, processes transactions with ML model, returns statistics
"""

import os
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sqlalchemy import create_engine

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "dbname": "Neobank_UNI",
    "user": "postgres",
    "password": os.environ.get("PASSWORD", "931579"),
    "host": "localhost",
    "port": "5432"
}

# Global variables
model = None
scaler = None
connected_clients = []
processing_active = False
metrics = {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "total_transactions": 0,
    "fraud_detected": 0,
    "fraud_rate": 0.0,
    "processing_time": 0.0,
    "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
}

class ProcessingStatus(BaseModel):
    status: str
    metrics: Dict
    is_processing: bool
    model_loaded: bool
    last_updated: Optional[str] = None

def load_model():
    """Load the trained fraud detection model"""
    global model, scaler
    try:
        model = joblib.load('logistic_regression_model.pkl')
        # Try to load scaler if available
        try:
            scaler = joblib.load('scaler.pkl')
        except:
            logger.warning("Scaler not found, will use manual scaling")
            scaler = None
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def get_db_connection():
    """Create a database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def fetch_transactions(limit: Optional[int] = None) -> pd.DataFrame:
    """Fetch transactions from fraud_data table using SQLAlchemy"""
    query = "SELECT * FROM fraud_data"
    if limit:
        query += f" LIMIT {limit}"
    
    try:
        # Create SQLAlchemy engine to fix pandas warning
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
        engine = create_engine(connection_string)
        
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        logger.error(f"Failed to fetch transactions: {e}")
        return pd.DataFrame()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for model prediction - data is already scaled in fraud_data table"""
    if df.empty:
        return df
    
    # Create a copy to avoid modifying original
    X = df.copy()
    
    # Expected columns from fraud_data table (already scaled)
    expected_cols = ['scaled_amount', 'scaled_time'] + [f'v{i}' for i in range(1, 29)]
    if not all(col in X.columns for col in expected_cols):
        logger.error(f"Missing expected columns. Found: {X.columns.tolist()}")
        logger.error(f"Expected: {expected_cols}")
        return pd.DataFrame()
    
    # Remove non-feature columns (keep id for reference but don't use in prediction)
    columns_to_drop = ['id', 'class']
    for col in columns_to_drop:
        if col in X.columns:
            X = X.drop(col, axis=1)
    
    # Data is already scaled in the fraud_data table, so no additional scaling needed
    # Order features exactly as they were during training
    # Based on model training: ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]
    # But our database uses lowercase v1-v28, so we need to map them to uppercase for the model
    
    # Convert lowercase v1-v28 to uppercase V1-V28 to match model training
    column_mapping = {}
    for i in range(1, 29):
        if f'v{i}' in X.columns:
            column_mapping[f'v{i}'] = f'V{i}'
    
    X = X.rename(columns=column_mapping)
    
    # Order features exactly as they were during training
    feature_cols = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]
    
    # Verify all required columns exist
    missing_cols = [col for col in feature_cols if col not in X.columns]
    if missing_cols:
        logger.error(f"Missing feature columns after preprocessing: {missing_cols}")
        logger.error(f"Available columns: {X.columns.tolist()}")
        return pd.DataFrame()
    
    # Return the properly ordered DataFrame
    return X[feature_cols]

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle case where confusion matrix might not be 2x2
    tn, fp, fn, tp = 0, 0, 0, 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        # Only one class present
        if y_true[0] == 0:
            tn = cm[0, 0]
        else:
            tp = cm[0, 0]
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

async def process_transactions():
    """Process transactions and update metrics - async version"""
    global metrics, processing_active
    
    if not model:
        logger.error("Model not loaded")
        return
    
    start_time = datetime.now()
    
    # Fetch transactions from fraud_data table
    df = fetch_transactions()
    if df.empty:
        logger.warning("No transactions found in fraud_data table")
        # Use mock data for demonstration
        df = generate_mock_data(1000)
    
    # Check if we have true labels
    has_labels = 'class' in df.columns
    y_true = df['class'].values if has_labels else None
    
    # Preprocess data
    X = preprocess_data(df)
    if X.empty:
        logger.error("Preprocessing failed")
        return
    
    # Make predictions
    try:
        # Convert to numpy array to match training format (fixes sklearn warning)
        X_array = X.values
        y_pred = model.predict(X_array)
        y_pred_proba = model.predict_proba(X_array)[:, 1]
        
        logger.info(f"Successfully processed {len(y_pred)} transactions")
        logger.info(f"Feature matrix shape: {X_array.shape}")
        logger.info(f"Sample predictions: {y_pred[:10]}")
        logger.info(f"Sample probabilities: {y_pred_proba[:10]}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(f"Feature matrix shape: {X.shape}")
        logger.error(f"Feature columns: {X.columns.tolist()}")
        return
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Update metrics
    total_transactions = len(y_pred)
    fraud_detected = int(np.sum(y_pred))
    fraud_rate = (fraud_detected / total_transactions * 100) if total_transactions > 0 else 0
    
    metrics.update({
        "total_transactions": total_transactions,
        "fraud_detected": fraud_detected,
        "fraud_rate": round(fraud_rate, 2),
        "processing_time": round(processing_time / total_transactions, 2) if total_transactions > 0 else 0
    })
    
    # If we have true labels, calculate accuracy metrics
    if y_true is not None:
        eval_metrics = evaluate_model(y_true, y_pred)
        
        # Debug: Log actual vs predicted distribution
        actual_fraud = int(np.sum(y_true))
        actual_normal = len(y_true) - actual_fraud
        logger.info(f"ACTUAL DISTRIBUTION - Normal: {actual_normal}, Fraud: {actual_fraud} ({actual_fraud/len(y_true)*100:.1f}%)")
        logger.info(f"PREDICTED DISTRIBUTION - Normal: {len(y_pred) - fraud_detected}, Fraud: {fraud_detected} ({fraud_rate:.1f}%)")
        
        metrics.update({
            "accuracy": round(eval_metrics["accuracy"] * 100, 2),
            "precision": round(eval_metrics["precision"] * 100, 2),
            "recall": round(eval_metrics["recall"] * 100, 2),
            "f1_score": round(eval_metrics["f1_score"] * 100, 2),
            "confusion_matrix": eval_metrics["confusion_matrix"]
        })
    else:
        # Without true labels, we can't calculate accuracy
        metrics.update({
            "accuracy": "--",
            "precision": "--",
            "recall": "--",
            "f1_score": "--"
        })
    
    logger.info(f"Processed {total_transactions} transactions, found {fraud_detected} fraudulent")

def generate_mock_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate mock transaction data matching fraud_data table schema"""
    np.random.seed(42)
    
    # Generate features to match fraud_data table structure (already scaled)
    data = {
        'scaled_time': np.random.randn(n_samples),      # Already scaled time
        'scaled_amount': np.random.randn(n_samples),    # Already scaled amount
        'id': range(1, n_samples + 1),                  # Sequential IDs
    }
    
    # Generate v1-v28 features (lowercase, normally distributed - already processed)
    for i in range(1, 29):
        data[f'v{i}'] = np.random.randn(n_samples)
    
    # Generate labels (5% fraud rate)
    data['class'] = np.random.binomial(1, 0.05, n_samples)
    
    return pd.DataFrame(data)

async def broadcast_metrics():
    """Broadcast metrics to all connected WebSocket clients"""
    if not connected_clients:
        return
    
    message = {
        "status": "Processing" if processing_active else "Idle",
        "is_processing": processing_active,
        "model_loaded": model is not None,
        "last_updated": datetime.now().isoformat(),
        **metrics
    }
    
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.append(client)
    
    # Remove disconnected clients
    for client in disconnected:
        connected_clients.remove(client)

async def continuous_processing():
    """Continuously process transactions while active"""
    global processing_active
    
    while processing_active:
        await process_transactions()
        await broadcast_metrics()
        await asyncio.sleep(2)  # Process every 2 seconds

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    load_model()
    logger.info("Server started")
    yield
    # Shutdown
    global processing_active
    processing_active = False
    logger.info("Server shutting down")

# FastAPI app with lifespan
app = FastAPI(title="Fraud Detection Server", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Fraud Detection Server", "model_loaded": model is not None}

@app.get("/status", response_model=ProcessingStatus)
async def get_status():
    """Get current processing status and metrics"""
    return ProcessingStatus(
        status="Processing" if processing_active else "Idle",
        metrics=metrics,
        is_processing=processing_active,
        model_loaded=model is not None,
        last_updated=datetime.now().isoformat()
    )

@app.post("/process")
async def process_once():
    """Process transactions once and return results"""
    if not model:
        return {"error": "Model not loaded"}
    
    await process_transactions()
    return {"status": "completed", "metrics": metrics}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    # Send initial status
    await websocket.send_json({
        "status": "Connected",
        "model_loaded": model is not None,
        **metrics
    })
    
    try:
        while True:
            # Receive commands from client
            data = await websocket.receive_json()
            command = data.get("command")
            
            global processing_active
            
            if command == "start_processing":
                if not processing_active:
                    processing_active = True
                    asyncio.create_task(continuous_processing())
                    await websocket.send_json({"status": "Processing started"})
            
            elif command == "stop_processing":
                processing_active = False
                await websocket.send_json({"status": "Processing stopped"})
            
            elif command == "get_status":
                await websocket.send_json({
                    "status": "Processing" if processing_active else "Idle",
                    "is_processing": processing_active,
                    "model_loaded": model is not None,
                    **metrics
                })
            
            elif command == "reset_metrics":
                metrics.update({
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "total_transactions": 0,
                    "fraud_detected": 0,
                    "fraud_rate": 0.0,
                    "processing_time": 0.0,
                    "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
                })
                await websocket.send_json({"status": "Metrics reset", **metrics})
    
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)