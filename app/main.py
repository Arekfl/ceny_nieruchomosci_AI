"""
FastAPI application for property price prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from datetime import datetime

from .models import (
    PredictionRequest, 
    PredictionResponse,
    FilterRequest,
    ModelInfo
)
from .config import (
    get_model,
    get_encoders,
    get_features,
    MODEL_METADATA,
    DATA_DIR
)

# Create FastAPI app
app = FastAPI(
    title="Property Price Prediction API",
    description="AI-powered system for predicting residential property prices in Poland",
    version="1.0.0",
    docs_url=None,
    redoc_url="/docs"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load models and data on startup"""
    try:
        model = get_model()
        encoders = get_encoders()
        features = get_features()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        raise


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Property Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - POST request to predict property price",
            "info": "/info - GET model information",
            "docs": "/docs - API documentation",
            "health": "/health - Health check"
        }
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/info", response_model=ModelInfo, tags=["Info"])
async def get_model_info():
    """Get information about the trained model"""
    return ModelInfo(
        model_type=MODEL_METADATA["model_type"],
        algorithm=MODEL_METADATA["algorithm"],
        features_used=MODEL_METADATA["features_used"],
        training_samples=MODEL_METADATA["training_samples"],
        test_r2_score=MODEL_METADATA["test_r2_score"],
        test_rmse=MODEL_METADATA["test_rmse"],
        test_mae=MODEL_METADATA["test_mae"],
        last_updated=MODEL_METADATA["last_updated"]
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(request: PredictionRequest):
    """
    Predict property price based on characteristics
    
    ### Example Request:
    ```json
    {
        "area": 120.5,
        "rooms": 4,
        "year_constructed": 2020,
        "heating": "gazowe",
        "building_material": "cegła",
        "building_type": "bliźniak",
        "market": "pierwotny",
        "voivodeship": "mazowieckie",
        "city": "Kraków"
    }
    ```
    """
    try:
        # Load model, encoders and features
        model = get_model()
        encoders = get_encoders()
        features = get_features()
        
        # Prepare input data
        feature_dict = {
            'Area (m²)': request.area,
            'Number of rooms': request.rooms,
            'year_const': request.year_constructed,
            'Heating': request.heating.value,
            'Building material': request.building_material.value,
            'Building type': request.building_type.value,
            'Market': request.market.value,
            'voivodeship': request.voivodeship.value,
        }
        
        # Encode categorical features
        X = pd.DataFrame([feature_dict])
        
        for col in ['Heating', 'Building material', 'Building type', 'Market', 'voivodeship']:
            if col in encoders:
                X[col] = encoders[col].transform(X[col])
        
        # Reorder columns to match training features
        X = X[features]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Calculate local statistics if city or district provided
        local_stats = None
        if request.city or request.district:
            try:
                df = pd.read_csv(DATA_DIR / "data_processed.csv")
                filtered_df = df.copy()
                
                if request.city:
                    filtered_df = filtered_df[filtered_df['city'].str.lower() == request.city.lower()]
                
                if request.district and len(filtered_df) > 0:
                    filtered_df = filtered_df[filtered_df['district'].str.lower() == request.district.lower()]
                
                if len(filtered_df) > 0:
                    local_stats = {
                        "location": {
                            "city": request.city,
                            "district": request.district
                        },
                        "properties_count": len(filtered_df),
                        "avg_price": round(float(filtered_df['Price'].mean()), 2),
                        "min_price": round(float(filtered_df['Price'].min()), 2),
                        "max_price": round(float(filtered_df['Price'].max()), 2),
                        "avg_area": round(float(filtered_df['Area (m²)'].mean()), 2),
                        "avg_rooms": round(float(filtered_df['Number of rooms'].mean()), 2),
                        "avg_year": int(filtered_df['year_const'].mean())
                    }
            except Exception as e:
                # If local stats fail, continue with prediction only
                print(f"Warning: Could not load local statistics: {e}")
        
        # Determine confidence based on input reasonableness
        confidence = determine_confidence(request, prediction)
        
        return PredictionResponse(
            predicted_price=round(float(prediction), 2),
            currency="PLN",
            confidence=confidence,
            input_features=request,
            local_stats=local_stats
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error making prediction: {str(e)}"
        )


@app.get("/filter", tags=["Info"])
async def filter_properties(
    voivodeship: str = None,
    city: str = None,
    district: str = None
):
    """
    Filter available properties by location
    
    Returns statistics about properties matching the filter criteria
    """
    try:
        # Load processed data
        df = pd.read_csv(DATA_DIR / "data_processed.csv")
        
        # Apply filters
        filtered_df = df.copy()
        
        if voivodeship:
            filtered_df = filtered_df[filtered_df['voivodeship'] == voivodeship]
        
        if city:
            filtered_df = filtered_df[filtered_df['city'] == city]
        
        if district:
            filtered_df = filtered_df[filtered_df['district'] == district]
        
        if len(filtered_df) == 0:
            raise HTTPException(
                status_code=404,
                detail="No properties found matching the filter criteria"
            )
        
        # Calculate statistics
        stats = {
            "count": len(filtered_df),
            "avg_price": round(float(filtered_df['Price'].mean()), 2),
            "min_price": round(float(filtered_df['Price'].min()), 2),
            "max_price": round(float(filtered_df['Price'].max()), 2),
            "avg_area": round(float(filtered_df['Area (m²)'].mean()), 2),
            "avg_rooms": round(float(filtered_df['Number of rooms'].mean()), 2),
            "filters_applied": {
                "voivodeship": voivodeship,
                "city": city,
                "district": district
            }
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error filtering properties: {str(e)}"
        )


def determine_confidence(request: PredictionRequest, prediction: float) -> str:
    """Determine confidence level based on input characteristics"""
    confidence_score = 0.0
    
    # Area between 40-200 m² is common
    if 40 <= request.area <= 200:
        confidence_score += 0.3
    
    # Rooms between 2-5 is common
    if 2 <= request.rooms <= 5:
        confidence_score += 0.2
    
    # Year between 1960-2025
    if 1960 <= request.year_constructed <= 2025:
        confidence_score += 0.2
    
    # Typical price range
    if 100000 <= prediction <= 1500000:
        confidence_score += 0.3
    
    if confidence_score >= 0.8:
        return "High"
    elif confidence_score >= 0.5:
        return "Medium"
    else:
        return "Low"


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return {
        "error": str(exc),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
