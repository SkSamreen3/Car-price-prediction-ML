# Car Price Prediction

An end-to-end machine learning project that predicts the selling price of used cars using regression models, served via a Streamlit web application.

## Project Structure

```
car_price/
├── config/
│   ├── __init__.py
│   └── config.py              # Centralized configuration & constants
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data loading, validation, encoding
│   ├── model.py               # Training, saving, loading models
│   ├── evaluate.py            # Metrics computation & model comparison
│   └── visualize.py           # Plotting utilities
├── models/                    # Saved model artifacts & plots
├── Data_Files/                # Raw CSV datasets
├── app.py                     # Streamlit web application
├── train.py                   # CLI training pipeline
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train all models and auto-select the best
python train.py

# Or train a specific model
python train.py --model RandomForest
```

This will:
- Load and preprocess `Data_Files/car data.csv`
- Train and compare 5 regression models
- Save the best model to `models/best_model.joblib`
- Save metrics to `models/metrics.json`
- Generate evaluation plots in `models/plots/`

### 3. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Models Compared

| Model               | Description                          |
|---------------------|--------------------------------------|
| Linear Regression   | Baseline OLS                         |
| Lasso               | L1 regularized regression            |
| Ridge               | L2 regularized regression            |
| Random Forest       | Ensemble of decision trees           |
| Gradient Boosting   | Sequential boosting ensemble         |

## Metrics Reported

- **R²** — Coefficient of determination
- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error

## Docker Deployment

```bash
docker build -t car-price-predictor .
docker run -p 8501:8501 car-price-predictor
```

## Dataset

The primary dataset (`car data.csv`) contains 301 entries with features:

| Feature        | Description                          |
|----------------|--------------------------------------|
| Car_Name       | Name of the car (dropped in training)|
| Year           | Manufacturing year                   |
| Selling_Price  | Price the owner wants to sell (target)|
| Present_Price  | Current ex-showroom price (Lakhs)    |
| Kms_Driven     | Kilometers driven                    |
| Fuel_Type      | Petrol / Diesel / CNG                |
| Seller_Type    | Dealer / Individual                  |
| Transmission   | Manual / Automatic                   |
| Owner          | Number of previous owners            |
