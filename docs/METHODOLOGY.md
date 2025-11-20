# Methodology

## 1. Data Pipeline
-   **Collection**:
    -   **Ground Truth**: PM2.5, PM10, NO2, etc., from CPCB/OpenAQ for Chennai (Urban).
    -   **Satellite**: AOD (MODIS/VIIRS), NO2/CO (Sentinel-5P) for both Urban and Rural regions.
    -   **Weather**: Temp, Humidity, Wind Speed from ERA5/OpenWeatherMap.
-   **Preprocessing**:
    -   **Spatial Matching**: k-NN matching of satellite pixels to ground coordinates.
    -   **Temporal Alignment**: Resampling to daily/hourly averages.
    -   **Imputation**: Linear interpolation for short gaps.
-   **Feature Engineering**:
    -   Lag features (1-day, 3-day).
    -   Rolling means.
    -   Seasonal indicators (Month, Day of Week).

## 2. Model Architecture
### Urban Model (Source Domain)
-   Trained on labeled urban data.
-   Algorithms: Linear Regression, Random Forest, XGBoost, LightGBM.
-   **Input**: Satellite AOD + Weather + Time features.
-   **Output**: Ground Pollutant Concentration.

### Transfer Learning (Domain Adaptation)
-   **Challenge**: Rural areas lack ground truth labels (y), but have abundant satellite data (X).
-   **Solution**: Domain Adversarial Neural Network (DANN).
    -   **Feature Extractor**: Maps X to a common latent space.
    -   **Label Predictor**: Predicts pollutant levels (trained on Urban).
    -   **Domain Classifier**: Tries to distinguish between Urban and Rural inputs.
    -   **Objective**: Minimize Label Loss - Adversarial Domain Loss. This forces the Feature Extractor to learn domain-invariant features.

## 3. AQI Calculation
-   Predicted concentrations are converted to AQI using the standard CPCB formula and breakpoints.

## 4. Visualization
-   **Streamlit Dashboard**:
    -   Interactive Maps (Folium).
    -   Time-series analysis.
    -   Comparative model performance.
