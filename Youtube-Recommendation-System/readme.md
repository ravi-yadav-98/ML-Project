# YouTube Recommendation System

This project implements a collaborative filtering-based recommendation system for YouTube videos. It uses a neural network model 
to predict user-item interactions and provides recommendations based on predicted ratings. The system includes a FastAPI endpoint for real-time predictions.

---

### Key Features:
- Collaborative filtering using a neural network.
- Training and evaluation on user-item interaction data.
- FastAPI endpoint for real-time predictions.
---

## Techniques Used in the Recommendation System

The project employs **collaborative filtering** with a **neural network-based approach** to recommend YouTube videos. Collaborative filtering leverages user-item interaction data to predict how a user might rate or interact with an item. The model uses **embedding layers** to represent users and items in a low-dimensional space, followed by **fully connected layers** to capture complex patterns in the data. The implementation uses PyTorch for building and training the neural network, with dropout and batch normalization for regularization.

---

### Project Directory Structure
```
youtube-recommendation-system/
│
├── ml-100k/                     # Dataset files
│   ├── u1.base
│   ├── u1.test
│
├── saved_model/                    # Trained Model files
│
├── data_loader.py            # Data loading and preprocessing
├── train.py                  # Training script
├── trainer.py                  # Trainer Code
├── recommend.py              # Prediction script
├── main.py                   # FastAPI application
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
```
---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/youtube-recommendation-system.git
cd youtube-recommendation-system
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```


## Training the model

### 1. Prepare the Dataset
Ensure your dataset files (`u{id_val}.base` and `u{id_val}.test`) are placed in the `ml-100k/` directory. The dataset should have the following columns:
- `user_id`: Unique identifier for each user.
- `item_id`: Unique identifier for each item (e.g., YouTube video).
- `rating`: Interaction strength between the user and the item.
- `ts`: Timestamp of the interaction (optional).

### 2. Update Configuration
Edit the `config.py` file to specify the following hyperparameters:
- `num_users`: Total number of users in the dataset.
- `num_items`: Total number of items in the dataset.
- `emb_size`: Size of the user and item embeddings.
- `emb_dropout`: Dropout rate for the embedding layers.
- `fc_layer_sizes`: List of sizes for the fully connected layers.
- `dropout`: List of dropout rates for each fully connected layer.
- `out_range`: Range of the output ratings (e.g., `[0, 5]`).

### 3. Run the Training Script
``` bash
python train.py
```
- After training, the best model will be saved as model/best_model.pth and loss curve will be saved in plots folder
- Adjust hyperparameters in config.py to optimize model performance.
- Use a GPU for faster training by setting device = torch.device("cuda:0") in the code.
---

## Testing the Model with FastAPI

Once the FastAPI server is running, you can test the model by sending requests to the `/predict` endpoint. Below are the steps to test the model:

### 1. Start the FastAPI Server
Run the following command to start the FastAPI server:
```bash
uvicorn main:app --reload
```
The server will be available at http://127.0.0.1:8000

### 2.  Test the API Using Swagger UI
- Open your browser and go to http://127.0.0.1:8000/docs.
- Click on the /predict endpoint.
- Click the Try it out button.
- Enter the input data in the request body field:
  ```json
  {
  "data": [
    {"user_id": 1, "item_id": 2},
    {"user_id": 1, "item_id": 3},
    {"user_id": 5, "item_id": 10}
  ]
  }
- Click the Execute button.
- View the predicted ratings in the response section.

### 3.  Test the API Using curl
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "data": [
    {"user_id": 1, "item_id": 2},
    {"user_id": 1, "item_id": 3},
    {"user_id": 5, "item_id": 10}
  ]
}'
```
Expected Response
```bash
[4.5, 3.8, 2.1]
```

## Contact
For questions, feedback, or contributions, feel free to reach out:

- **Ravi Prakash Yadav**  
- **Email**: raviprakashyadav1998.com  
- **GitHub**: [raviyadav-98](https://github.com/ravi-yadav-98)  
- **LinkedIn**: [Ravi Prakash Yadav](https://www.linkedin.com/in/raviyadav98/)   

---

Thank you for your interest in this project! 🚀
