# 🐦 Bird Strike Analysis & Prediction System

<img width="1911" height="852" alt="Bird_app" src="https://github.com/user-attachments/assets/9229b30e-16fe-4313-8046-e83f6213986b" />


## About This Project
 a Streamlit-powered dashboard that:

- 📊 Analyzes historical bird strike data
- 🤖 Predicts aircraft damage risk using XGBoost
- 📈 Generates interactive visualizations
- ✈️ Helps aviation professionals mitigate wildlife hazards

**Key Features:**
- Data exploration tools
- Machine learning modeling
- 12+ interactive charts
- Model interpretability (Permutation importance)

Built with: `Python` `Streamlit` `Pandas` `XGBoost` `Plotly`

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
