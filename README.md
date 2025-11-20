# Granular Rural Air Quality Estimation Using Transfer Learning

## Problem Statement
Air quality monitoring in India is heavily skewed towards urban centers, leaving rural areas with little to no ground-level data. This project aims to estimate rural air quality (AQI) by leveraging satellite data and transferring knowledge from urban models using Domain Adaptation techniques.

## Objectives
1.  **Data Integration**: Combine ground station data (Urban) with satellite aerosol products (MODIS, VIIRS, Sentinel-5P) and meteorological data.
2.  **Model Development**: Train robust ML models on urban data.
3.  **Domain Adaptation**: Adapt urban models to rural settings using DANN (Domain Adversarial Neural Networks) to handle distribution shifts.
4.  **Visualization**: Provide a granular, interactive dashboard for monitoring rural AQI.

## Project Structure
```
ML Project/
├── config/             # Configuration files
├── data/               # Data storage (raw/processed)
├── docs/               # Documentation
├── src/                # Source code
│   ├── data_collection/
│   ├── preprocessing/
│   ├── models/
│   ├── dashboard/
│   └── utils/
├── main.py             # End-to-end execution script
└── requirements.txt    # Dependencies
```

## Setup and Usage
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Data Pipeline & Training**:
    ```bash
    python main.py
    ```
3.  **Launch Dashboard**:
    ```bash
    streamlit run src/dashboard/app.py
    ```

## Key Technologies
-   **Data**: OpenAQ, NASA Earthdata, Open-Meteo
-   **ML**: Scikit-learn, XGBoost, LightGBM, PyTorch (for DANN)
-   **Geospatial**: Rasterio, GDAL, Folium
-   **App**: Streamlit
