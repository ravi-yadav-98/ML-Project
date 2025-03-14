# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import recommend  # Import the recommend function
import config

# Initialize FastAPI app
app = FastAPI()

# Define the request body schema
class UserItemPair(BaseModel):
    user_id: int
    item_id: int

class PredictionRequest(BaseModel):
    data: List[UserItemPair]

# Load the model path and config
MODEL_PATH = "saved_models/best_model.pth"
CONFIG = config.config

# Define the prediction endpoint
@app.post("/predict", response_model=List[float])
def predict(request: PredictionRequest):
    """
    Predict ratings for a list of user-item pairs.
    
    Args:
        request (PredictionRequest): List of user-item pairs.
    
    Returns:
        List[float]: Predicted ratings for the input pairs.
    """
    try:
        # Convert the request data to the format expected by the recommend function
        new_data = [{"user_id": pair.user_id, "item_id": pair.item_id} for pair in request.data]
        
        # Get predictions
        predictions = recommend.recommend(MODEL_PATH, new_data, CONFIG)
        
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)