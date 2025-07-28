import os
os.system("pip install matplotlib")
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance
import io
from tqdm import tqdm
# Then wrap your training loop with tqdm

# Set page config
st.set_page_config(
    page_title="Bird Strike Analysis & Prediction",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with more robust styling
st.markdown("""
<style>
    .main {
        background-color: black;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stFileUploader, .stSlider {
        margin-bottom: 1rem;
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stAlert {
        border-radius: 8px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# App title with better spacing
st.title("🐦 Bird Strike Analysis & Prediction System")
st.markdown("---")

# Sidebar for navigation with improved styling
with st.sidebar:
    st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        .sidebar .sidebar-content .stRadio > label {
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("Navigation")
    app_mode = st.radio("Choose a section:", 
                       ["📊 Data Analysis", "🤖 Prediction Model", "📈 Visualizations"],
                       label_visibility="collapsed")
    
    st.markdown("---")
    st.header("About")
    st.markdown("""
    <div style="color: black;">
    This app analyzes bird strike data and predicts potential aircraft damage.
    - *Data Analysis*: Explore the dataset
    - *Prediction Model*: Train and evaluate XGBoost model
    - *Visualizations*: Interactive charts and graphs
    </div>
    """, unsafe_allow_html=True)

# Load data function with better caching and error handling
@st.cache_data(show_spinner="Loading data...")
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            # Try sample data if no file uploaded
            sample_data = {
                'RecordID': [1, 2, 3],
                'FlightDate': ['2020-01-01', '2020-01-02', '2020-01-03'],
                'Damage': ['No Damage', 'Caused Damage', 'No Damage'],
                'Altitude': [1000, 2000, 1500],
                'WildlifeSize': ['Small', 'Medium', 'Large'],
                'FlightPhase': ['Approach', 'Take-off', 'Climb']
            }
            data = pd.DataFrame(sample_data)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Data Analysis Section
if app_mode == "📊 Data Analysis":
    st.header("📊 Bird Strike Data Analysis")
    
    # Initialize session state if not exists
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # File uploader in data analysis section too
    uploaded_file = st.file_uploader("Upload your bird strike data (CSV format)", type="csv")
    
    if uploaded_file:
        st.session_state.data = load_data(uploaded_file)
    
    if st.session_state.data is not None:
        data = st.session_state.data
        st.success(f"Dataset loaded successfully with {len(data)} records!")
        
        # Show basic info in columns
        st.subheader("Dataset Overview")
        cols = st.columns(4)
        cols[0].metric("Total Records", len(data))
        cols[1].metric("Columns", len(data.columns))
        cols[2].metric("Missing Values", data.isnull().sum().sum())
        cols[3].metric("Damage Rate", 
                       f"{data['Damage'].value_counts(normalize=True).get('Caused Damage', 0)*100:.1f}%"
                       if 'Damage' in data.columns else "N/A")
        
        # Show data with expander
        with st.expander("Show raw data"):
            st.dataframe(data.head(100))
        
        # Missing data analysis
        st.subheader("Missing Data Analysis")
        missing_df = data.isnull().sum().rename('missing').reset_index()
        missing_df.columns = ['Column', 'Missing Values']
        missing_df['Missing %'] = (missing_df['Missing Values']/len(data)*100).round(1)
        missing_df = missing_df[missing_df['Missing Values'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df.style.bar(subset='Missing %', color='#ff6b6b'), 
                         use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
        
        # Basic statistics with tabs
        st.subheader("Basic Statistics")
        tab1, tab2 = st.tabs(["Numerical", "Categorical"])
        
        with tab1:
            num_cols = data.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                st.dataframe(data[num_cols].describe(), use_container_width=True)
            else:
                st.info("No numerical columns found")
        
        with tab2:
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                st.dataframe(data[cat_cols].describe(), use_container_width=True)
            else:
                st.info("No categorical columns found")

# Prediction Model Section
elif app_mode == "🤖 Prediction Model":
    st.header("🤖 Damage Prediction Model")
    
    # File uploader with better feedback
    uploaded_file = st.file_uploader("Upload your bird strike data (CSV format)", type="csv",
                                    help="Please upload a CSV file with bird strike records")
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.session_state.data = data
        st.success(f"Dataset loaded successfully with {len(data)} records!")
        
        # Data preprocessing section
        st.subheader("Data Preprocessing")
        
        # Create binary target with checks
        if 'Damage' not in data.columns:
            st.error("The dataset must contain a 'Damage' column for prediction")
        else:
            data['DamageBinary'] = data['Damage'].apply(
                lambda x: 1 if str(x).strip().lower() in ['caused damage', 'yes', 'true', '1'] else 0
            )
            
            # Show target distribution with columns
            st.write("**Target Variable Distribution**")
            col1, col2 = st.columns(2)
            
            with col1:
                target_dist = data['DamageBinary'].value_counts(normalize=True) * 100
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(target_dist, labels=['No Damage', 'Damage'], autopct='%1.1f%%', 
                       colors=['#4CAF50', '#F44336'], startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            
            with col2:
                st.metric("Damage Cases", f"{target_dist[1]:.1f}%")
                st.metric("No Damage Cases", f"{target_dist[0]:.1f}%")
                st.caption("Note: Damage is defined as any impact that caused damage to the aircraft")
            
            # Feature selection with better defaults
            st.subheader("Feature Selection")
            all_cols = data.columns.tolist()
            default_drop = ['RecordID', 'FlightDate', 'Remarks', 'Cost', 'NumberStruck', 
                           'WildlifeSpecies', 'AirportName', 'Damage']
            available_drop = [col for col in default_drop if col in all_cols]
            
            drop_cols = st.multiselect(
                "Select columns to exclude from modeling:", 
                all_cols, 
                default=available_drop,
                help="Remove columns that won't be useful for prediction"
            )
            
            # Model parameters with better organization
            st.subheader("Model Parameters")
            with st.form("model_params"):
                col1, col2, col3 = st.columns(3)
                test_size = col1.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05,
                                      help="Percentage of data to use for testing")
                random_state = col2.number_input("Random Seed", 0, 100, 42,
                                               help="Seed for reproducible results")
                n_estimators = col3.selectbox("Number of Trees", [50, 100, 200, 300], index=1,
                                            help="More trees can improve performance but increase training time")
                
                scale_pos_weight = st.slider("Class Weight (Damage)", 1, 10, 3,
                                           help="Adjust for class imbalance (higher values give more weight to damage cases)")
                
                submitted = st.form_submit_button("Train XGBoost Model")
                
            if submitted:
                with st.spinner('Training model... This may take a few minutes'):
                    try:
                        # Drop selected columns
                        df = data.drop(columns=drop_cols)
                        
                        # Handle missing and categorical data
                        df.fillna('Unknown', inplace=True)
                        label_encoders = {}
                        for col in df.select_dtypes(include=['object', 'bool']).columns:
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                            label_encoders[col] = le
                        
                        # Split features and target
                        X = df.drop(columns=['DamageBinary'])
                        y = df['DamageBinary']
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, stratify=y, test_size=test_size, random_state=random_state
                        )
                        
                        # Train model with progress bar
                        progress_bar = st.progress(0)
                        model = XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=5,
                            learning_rate=0.1,
                            scale_pos_weight=scale_pos_weight,
                            random_state=random_state,
                            use_label_encoder=False,
                            eval_metric='logloss'
                        )
                        
                        # Simulate progress (in real app, you'd update during training)
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            time.sleep(0.01)  # Simulate work
                        
                        model.fit(X_train, y_train)
                        progress_bar.empty()
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Store model and data in session state
                        st.session_state.model = model
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.y_pred = y_pred
                        st.session_state.y_pred_proba = y_pred_proba
                        
                        # Show success message
                        st.success("Model trained successfully!")
                        
                        # Metrics section with tabs
                        st.subheader("Model Performance")
                        tab1, tab2, tab3 = st.tabs(["Classification Report", "Confusion Matrix", "ROC Curve"])
                        
                        with tab1:
                            st.write("**Classification Report**")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                        
                        with tab2:
                            st.write("**Confusion Matrix**")
                            conf_matrix = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                                        xticklabels=["No Damage", "Damage"], 
                                        yticklabels=["No Damage", "Damage"], ax=ax)
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            st.pyplot(fig)
                        
                        with tab3:
                            st.write("**ROC Curve**")
                            from sklearn.metrics import roc_curve, auc
                            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                            roc_auc = auc(fpr, tpr)
                            
                            fig, ax = plt.subplots()
                            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                                    label=f'ROC curve (area = {roc_auc:.2f})')
                            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            ax.set_xlim([0.0, 1.0])
                            ax.set_ylim([0.0, 1.05])
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('Receiver Operating Characteristic')
                            ax.legend(loc="lower right")
                            st.pyplot(fig)
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plot_importance(model, max_num_features=15, importance_type='gain', ax=ax)
                        st.pyplot(fig)
                        
                        # Download predictions
                        st.subheader("Download Predictions")
                        pred_df = X_test.copy()
                        pred_df['Actual'] = y_test
                        pred_df['Predicted'] = y_pred
                        pred_df['Prediction_Probability'] = y_pred_proba
                        
                        csv = pred_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="bird_strike_predictions.csv",
                            mime="text/csv",
                            help="Download the test set with model predictions"
                        )
                        
                    except Exception as e:
                        st.error(f"An error occurred during model training: {str(e)}")

# Visualizations Section
elif app_mode == "📈 Visualizations":
    st.header("📈 Interactive Visualizations")
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload a dataset in the Prediction Model section first.")
        st.info("You can use the sample data to explore visualization options")
        sample_data = {
            'FlightDate': pd.date_range('2010-01-01', '2020-12-31', freq='M'),
            'Altitude': np.random.randint(100, 10000, 132),
            'WildlifeSize': np.random.choice(['Small', 'Medium', 'Large'], 132),
            'WildlifeSpecies': np.random.choice(['European starling', 'Rock pigeon', 'Canada goose', 
                                               'Mourning dove', 'Mallard', 'Unknown bird - small'], 132),
            'FlightPhase': np.random.choice(['Take-off', 'Climb', 'Approach', 'Landing'], 132),
            'Damage': np.random.choice(['Caused Damage', 'No Damage'], 132, p=[0.2, 0.8]),
            'NumberStruckActual': np.random.poisson(2, 132)
        }
        st.session_state.data = pd.DataFrame(sample_data)
    
    data = st.session_state.data
    
    # Clean data for visualizations
    data['FlightDate'] = pd.to_datetime(data['FlightDate'])
    data['Year'] = data['FlightDate'].dt.year
    data['Month'] = data['FlightDate'].dt.month_name()
    
    # Visualization selection with description
    viz_option = st.selectbox(
        "Choose a visualization:",
        [
            "Collisions Over Years",
            "Bird Fatalities Over Years",
            "Cost of Collisions Over Years",
            "Fatalities by Bird Size",
            "Most Affected Small Bird Species",
            "Most Affected Medium Bird Species",
            "Most Affected Large Bird Species",
            "Fatalities by Flight Phase",
            "Fatalities by Altitude",
            "Damage by Bird Size",
            "Monthly Patterns",
            "Fatalities by State and Year",
            "Numeric Features Distribution"
        ],
        help="Select a visualization to explore different aspects of the data"
    )
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Create visualizations based on selection
    with st.container():
        st.markdown(f"<div class='plot-container'>", unsafe_allow_html=True)
        
        if viz_option == "Collisions Over Years":
            st.subheader("Total Number of Collisions Over the Years")
            agg_data = data['Year'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = sns.barplot(x=agg_data.index, y=agg_data, ax=ax, palette="rocket_r")
            bars.bar_label(bars.containers[0], fontsize=10, label_type='edge')
            
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of Collisions")
            ax.set_title("Bird Strike Incidents by Year", pad=20)
            st.pyplot(fig)
            
        elif viz_option == "Bird Fatalities Over Years":
            st.subheader("Total Bird Fatalities Over the Years")
            if 'NumberStruckActual' in data.columns:
                agg_data = data.groupby('Year')['NumberStruckActual'].sum()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                line = sns.lineplot(x=agg_data.index, y=agg_data, ax=ax, 
                                   color='#FF5722', linewidth=2.5, marker='o')
                
                # Annotate each point
                for x, y in zip(agg_data.index, agg_data):
                    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", 
                               xytext=(0,10), ha='center')
                
                ax.set_xlabel("Year")
                ax.set_ylabel("Total Fatalities")
                ax.set_title("Bird Fatalities by Year", pad=20)
                st.pyplot(fig)
            else:
                st.warning("This visualization requires 'NumberStruckActual' column in the data")
                
        elif viz_option == "Cost of Collisions Over Years":
            st.subheader("Total Cost of Collisions Over the Years")
            if 'Cost' in data.columns and 'Year' in data.columns:
                try:
                    # Clean cost data
                    data['Cost'] = pd.to_numeric(data['Cost'].astype(str).str.replace(',', ''), errors='coerce')
                    agg_data = data.groupby('Year')['Cost'].sum() / 1e6  # Convert to millions
            
                    if not agg_data.empty:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = sns.barplot(x=agg_data.index, y=agg_data, ax=ax, palette="rocket_r")
                
                         # Only add labels if bars exist and data is valid
                        if hasattr(bars, 'containers') and len(bars.containers) > 0:
                            try:
                                bars.bar_label(bars.containers[0], 
                                     fontsize=10, 
                                     label_type='edge',
                                     labels=[f"${x:.1f}M" for x in agg_data])
                            except Exception as e:
                                st.warning(f"Couldn't add bar labels: {str(e)}")
                
                        ax.set_xlabel("Year")
                        ax.set_ylabel("Cost (Millions USD)")
                        ax.set_title("Estimated Damage Costs by Year", pad=20)
                        st.pyplot(fig)
                    else:
                        st.warning("No valid cost data available for visualization")
                except Exception as e:
                    st.error(f"Error processing cost data: {str(e)}")
            else:
                st.warning("Required columns ('Cost' and 'Year') not found in data")
                 
        elif viz_option == "Fatalities by Bird Size":
            st.subheader("Fatalities by Bird Size")
            
            if 'WildlifeSize' in data.columns and 'NumberStruckActual' in data.columns:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Pie chart
                agg_pie = data.groupby('WildlifeSize').size().sort_values()
                ax1.pie(agg_pie, autopct='%.1f%%', labels=agg_pie.index,
                       colors=sns.color_palette("rocket_r", len(agg_pie)),
                       startangle=90)
                ax1.set_title('Incidents by Bird Size')
                
                # Bar chart
                agg_bar = data.groupby('WildlifeSize')['NumberStruckActual'].sum()
                sns.barplot(x=agg_bar.index, y=agg_bar, ax=ax2, 
                            palette="rocket_r")
                ax2.bar_label(ax2.containers[0], fontsize=10)
                ax2.set_title('Fatalities by Bird Size')
                ax2.set_ylabel('Total Fatalities')
                ax2.tick_params(axis='x', rotation=45)
                
                st.pyplot(fig)
            else:
                st.warning("This visualization requires 'WildlifeSize' and 'NumberStruckActual' columns")
        
        elif viz_option == "Most Affected Small Bird Species":
            st.subheader("Most Affected Small Bird Species")
    
            # Check required columns exist
            required_cols = ['WildlifeSize', 'WildlifeSpecies', 'NumberStruckActual']
            if all(col in data.columns for col in required_cols):
                try:
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 6))
            
                    # Prepare data - filter small birds and get top 10 species
                    small_birds = data[data['WildlifeSize'].str.strip().str.lower() == 'small']
                    agg_data = small_birds.groupby('WildlifeSpecies')['NumberStruckActual'].sum()
                    agg_data = agg_data.sort_values(ascending=False)[1:11]  # Skip top 1 if needed

                    if not agg_data.empty:
                        # Create colormap
                        cmap = plt.cm.get_cmap('rocket_r')
                        norm = plt.Normalize(agg_data.min(), agg_data.max())
                
                        # Create horizontal bar plot
                        bars = sns.barplot(
                            y=agg_data.index,
                            x=agg_data,
                            ax=ax,
                            orient='h',
                            palette='rocket_r'
                        )
                
                        # Add bar labels if bars exist
                        if bars.containers:
                            bars.bar_label(
                                bars.containers[0],
                                fontsize=12,
                                label_type='center',
                                fmt='%g'
                            )
                
                        # Color customization
                        for bar in bars.patches:
                            width = bar.get_width()
                            bar.set_facecolor(cmap(norm(width)*0.75))
                
                        # Formatting
                        plt.title('Top 10 Most Affected Small Bird Species', fontsize=14)
                        plt.xticks([])
                        plt.ylabel('Species')
                        plt.xlabel('')
                        plt.tight_layout()
                
                        # Display in Streamlit
                        st.pyplot(fig)
                
                        # Show data table
                        with st.expander("View data table"):
                            st.dataframe(
                                agg_data.reset_index().rename(columns={
                                    'WildlifeSpecies': 'Species',
                                    'NumberStruckActual': 'Strikes Count'
                                }).style.background_gradient(
                                    cmap='rocket_r',
                                    subset=['Strikes Count']
                                )
                            )
                    else:
                        st.warning("No data available for small bird species")
                
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
            else:
                st.warning(f"Required columns ({', '.join(required_cols)}) not found in data")

        elif viz_option == "Most Affected Medium Bird Species":
            st.subheader("Most Affected Medium Bird Species")
    
            # Check required columns exist
            required_cols = ['WildlifeSize', 'WildlifeSpecies', 'NumberStruckActual']
            if all(col in data.columns for col in required_cols):
                try:
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 6))
            
                    # Prepare data - filter medium birds and get top 10 species
                    medium_birds = data[data['WildlifeSize'].str.strip().str.lower() == 'medium']
                    agg_data = medium_birds.groupby('WildlifeSpecies')['NumberStruckActual'].sum()
                    agg_data = agg_data.sort_values(ascending=False)[:10]

                    if not agg_data.empty:
                        # Create colormap
                        cmap = plt.cm.get_cmap('rocket_r')
                        norm = plt.Normalize(agg_data.min(), agg_data.max())
                
                        # Create horizontal bar plot
                        bars = sns.barplot(
                            y=agg_data.index,
                            x=agg_data,
                            ax=ax,
                            orient='h',
                            palette='rocket_r'
                        )
                
                        # Add bar labels if bars exist
                        if bars.containers:
                            bars.bar_label(
                                bars.containers[0],
                                fontsize=12,
                                label_type='center',
                                fmt='%g'
                            )
                
                        # Color customization
                        for bar in bars.patches:
                            width = bar.get_width()
                            bar.set_facecolor(cmap(norm(width)*0.75))
                
                        # Formatting
                        plt.title('Top 10 Most Affected Medium Bird Species', fontsize=14)
                        plt.xticks([])
                        plt.ylabel('Species')
                        plt.xlabel('')
                        plt.tight_layout()
                
                        # Display in Streamlit
                        st.pyplot(fig)
                
                        # Show data table
                        with st.expander("View data table"):
                            st.dataframe(
                                agg_data.reset_index().rename(columns={
                                    'WildlifeSpecies': 'Species',
                                    'NumberStruckActual': 'Strikes Count'
                                }).style.background_gradient(
                                    cmap='rocket_r',
                                    subset=['Strikes Count']
                                )
                            )
                    else:
                        st.warning("No data available for medium bird species")
                
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
            else:
                st.warning(f"Required columns ({', '.join(required_cols)}) not found in data")

        elif viz_option == "Most Affected Large Bird Species":
            st.subheader("Most Affected Large Bird Species")
    
            # Check required columns exist
            required_cols = ['WildlifeSize', 'WildlifeSpecies', 'NumberStruckActual']
            if all(col in data.columns for col in required_cols):
                try:
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 6))
            
                    # Prepare data - filter large birds and get top 10 species
                    large_birds = data[data['WildlifeSize'].str.strip().str.lower() == 'large']
                    agg_data = large_birds.groupby('WildlifeSpecies')['NumberStruckActual'].sum()
                    agg_data = agg_data.sort_values(ascending=False)[:10]

                    if not agg_data.empty:
                        # Create colormap
                        cmap = plt.cm.get_cmap('rocket_r')
                        norm = plt.Normalize(agg_data.min(), agg_data.max())
                
                        # Create horizontal bar plot
                        bars = sns.barplot(
                            y=agg_data.index,
                            x=agg_data,
                            ax=ax,
                            orient='h',
                            palette='rocket_r'
                        )
                
                        # Add bar labels if bars exist
                        if bars.containers:
                            bars.bar_label(
                                bars.containers[0],
                                fontsize=12,
                                label_type='center',
                                fmt='%g'
                            )
                
                        # Color customization
                        for bar in bars.patches:
                            width = bar.get_width()
                            bar.set_facecolor(cmap(norm(width)*0.75))
                
                        # Formatting
                        plt.title('Top 10 Most Affected Large Bird Species', fontsize=14)
                        plt.xticks([])
                        plt.ylabel('Species')
                        plt.xlabel('')
                        plt.tight_layout()
                
                        # Display in Streamlit
                        st.pyplot(fig)
                
                        # Show data table
                        with st.expander("View data table"):
                            st.dataframe(
                                agg_data.reset_index().rename(columns={
                                    'WildlifeSpecies': 'Species',
                                    'NumberStruckActual': 'Strikes Count'
                                }).style.background_gradient(
                                    cmap='rocket_r',
                                    subset=['Strikes Count']
                                )
                            )
                    else:
                        st.warning("No data available for large bird species")
                
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
            else:
                st.warning(f"Required columns ({', '.join(required_cols)}) not found in data")
                
        elif viz_option == "Fatalities by Flight Phase":
            st.subheader("Fatalities by Flight Phase")
            
            if 'FlightPhase' in data.columns:
                # Clean flight phase data
                data['FlightPhase'] = data['FlightPhase'].replace({
                    'Descent': 'Others',
                    'Taxi': 'Others',
                    'Parked': 'Others',
                    'Unknown': 'Others',
                    np.nan: 'Others'
                })
                
                agg_data = data['FlightPhase'].value_counts()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(x=agg_data.index, y=agg_data, ax=ax, palette="rocket_r")
                bars.bar_label(bars.containers[0], fontsize=10)
                
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_xlabel("Flight Phase")
                ax.set_ylabel("Number of Incidents")
                ax.set_title("Incidents by Flight Phase", pad=20)
                st.pyplot(fig)
            else:
                st.warning("This visualization requires 'FlightPhase' column in the data")
                
        elif viz_option == "Fatalities by Altitude":
            st.subheader("Fatalities by Altitude")
            
            if 'Altitude' in data.columns:
                # Clean altitude data
                alt_data = data[data['Altitude'] > 0].copy()
                alt_data['AltitudeBin'] = pd.cut(alt_data['Altitude'], 
                                               bins=[0, 500, 1000, 2000, 5000, 10000, 20000],
                                               labels=['0-500', '500-1k', '1k-2k', '2k-5k', '5k-10k', '10k+'])
                
                agg_data = alt_data.groupby('AltitudeBin').size()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = sns.barplot(x=agg_data.index, y=agg_data, ax=ax, palette="rocket_r")
                bars.bar_label(bars.containers[0], fontsize=10)
                
                ax.set_xlabel("Altitude Bins (feet)")
                ax.set_ylabel("Number of Incidents")
                ax.set_title("Incidents by Altitude Range", pad=20)
                st.pyplot(fig)
            else:
                st.warning("This visualization requires 'Altitude' column in the data")
        
        elif viz_option == "Damage by Bird Size":
            st.subheader("Damage Probability by Bird Size")
    
            if 'WildlifeSize' in data.columns and 'DamageBinary' in data.columns:
                # Calculate damage probability
                prob_data = data.groupby('WildlifeSize')['DamageBinary'].mean().sort_values(ascending=False)
        
                # Data validation
                if prob_data.empty:
                    st.warning("No data available for visualization")
                elif all(prob_data == 0):
                    st.info("No damage incidents recorded for these parameters")
                else:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = sns.barplot(x=prob_data.index, y=prob_data, ax=ax, palette="rocket_r")
            
                    # Safe label adding
                    if bars.containers:
                        try:
                            bars.bar_label(bars.containers[0],
                                 labels=[f"{x*100:.1f}%" for x in prob_data],
                                 fontsize=10)
                        except Exception as e:
                            st.warning(f"Couldn't add labels: {str(e)}")
            
                    ax.set_xlabel("Bird Size")
                    ax.set_ylabel("Damage Probability")
                    ax.set_title("Probability of Damage by Bird Size", pad=20)
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
            else:
               st.warning("Required columns ('WildlifeSize' and 'DamageBinary') not found in data")

        elif viz_option == "Monthly Patterns":
            st.subheader("Monthly Patterns in Bird Strikes")
            
            if 'Month' in data.columns:
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                             'July', 'August', 'September', 'October', 'November', 'December']
                
                agg_data = data['Month'].value_counts()
                agg_data = agg_data.reindex(month_order, fill_value=0)
                
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.lineplot(x=agg_data.index, y=agg_data, ax=ax, 
                            color='#4CAF50', linewidth=2.5, marker='o')
                
                ax.set_xlabel("Month")
                ax.set_ylabel("Number of Incidents")
                ax.set_title("Monthly Pattern of Bird Strikes", pad=20)
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
            else:
                st.warning("This visualization requires date information to extract months")
                
        elif viz_option == "Fatalities by State and Year":
            st.subheader("Most Bird Fatalities by State Over the Years")
    
            # Check required columns exist
            if all(col in data.columns for col in ['OriginState', 'FlightDate', 'NumberStruckActual']):
                try:
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 8))
            
                    # Prepare data
                    data['FlightDate'] = pd.to_datetime(data['FlightDate'])
                    agg_data = data.groupby(['OriginState', data.FlightDate.dt.year])['NumberStruckActual'].sum().unstack()
            
                    # Get top 10 states
                    idx = agg_data.sum(axis=1).sort_values(ascending=False).index[:10]
            
                    # Create heatmap
                    sns.heatmap(
                        agg_data.loc[idx],
                        ax=ax,
                        annot=True,
                        cmap='rocket_r',
                        cbar=False,
                        fmt='g',
                        linewidths=.5
                    )
            
                   # Formatting
                    plt.title('Top 10 States by Bird Fatalities Over Years', fontsize=14)
                    plt.xticks(rotation=45, ha='right')
                    plt.xlabel('Year')
                    plt.ylabel('State')
                    plt.tight_layout()
            
                    # Display in Streamlit
                    st.pyplot(fig)
            
                    # Optional: Show raw data
                    with st.expander("Show raw data"):
                        st.dataframe(agg_data.loc[idx].style.background_gradient(cmap='rocket_r'))
                
                except Exception as e:
                    st.error(f"Error generating heatmap: {str(e)}")
            else:
                st.warning("Required columns (OriginState, FlightDate, NumberStruckActual) not found in data")
                
        elif viz_option == "Numeric Features Distribution":
            st.subheader("Distribution of Numeric Features")
            
            # Get numeric columns
            num_cols = data.select_dtypes(include=np.number).columns
            
            if len(num_cols) > 0:
                # Determine grid size
                n_cols = 2
                n_rows = int(np.ceil(len(num_cols) / n_cols))
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
                axes = axes.flatten()
                
                # Set color palette
                sns.set_palette('Set1')
                
                # Plot each numeric feature
                for i, col in enumerate(num_cols):
                    try:
                        if len(data[col].dropna()) > 0:  # Check if there's data to plot
                            sns.histplot(data[col], ax=axes[i], kde=True)
                            axes[i].set_yscale('log')
                            axes[i].set_title(f"Distribution of {col}")
                            axes[i].set_xlabel("")
                        else:
                            axes[i].text(0.5, 0.5, f"No data for {col}", 
                                       ha='center', va='center')
                            axes[i].set_title(f"Distribution of {col}")
                    except Exception as e:
                        axes[i].text(0.5, 0.5, f"Error plotting {col}", 
                                   ha='center', va='center')
                        axes[i].set_title(f"Distribution of {col}")
                
                # Hide any empty subplots
                for j in range(i+1, len(axes)):
                    fig.delaxes(axes[j])
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show summary statistics
                with st.expander("Show summary statistics"):
                    st.dataframe(data[num_cols].describe().T.style.background_gradient(
                        cmap='rocket_r',
                        subset=['mean', '50%', 'std']
                    ))
            else:
                st.warning("No numeric columns found in the dataset")
                
        st.markdown("</div>", unsafe_allow_html=True)
