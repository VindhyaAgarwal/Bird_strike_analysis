# 🐦 Bird Strike Analysis & Prediction System

<img width="1911" height="852" alt="Bird_app" src="https://github.com/user-attachments/assets/9229b30e-16fe-4313-8046-e83f6213986b" />


A Streamlit application for analyzing historical bird strike data and predicting potential aircraft damage using machine learning.

## Features

- Interactive data exploration
- Damage prediction using XGBoost
- Visualization dashboard
- Model performance metrics

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bird-strike-analysis.git
   cd bird-strike-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload your bird strike data (CSV format)
2. Explore data visualizations
3. Train the prediction model
4. View performance metrics

## Data Requirements

The app expects CSV data with columns like:
- `FlightDate`
- `Altitude`
- `WildlifeSize`
- `Damage` (target variable)
- etc.

## License
MIT
