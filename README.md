# NYC Taxi Typology & Anomaly Detection

This project analyzes NYC Yellow Taxi trip records using exploratory data analysis, dimensionality reduction, clustering and unsupervised anomaly detection.

The main goal is to understand typical taxi trip patterns and identify trips that are statistically unusual with respect to distance, duration, cost, route type and additional fees.

## Tech Stack

* **Language**: Python 3.x
* **Analysis**: Pandas, Scikit-learn, UMAP-learn
* **Visualization**: Altair

## Project Overview

The project focuses on one month of NYC Yellow Taxi data. The analysis is performed mainly in the Jupyter notebook:

```text
notebooks/start.ipynb
```

The notebook presents the full workflow step by step:

1. loading and inspecting the taxi dataset,
2. selecting useful columns and optimizing memory usage,
3. cleaning obvious logical errors,
4. creating basic trip features,
5. sampling data for interactive analysis,
6. extending the feature set with additional engineered features,
7. applying PCA and UMAP for dimensionality reduction,
8. applying KMeans clustering to find trip groups,
9. applying anomaly detection methods,
10. interpreting clusters, anomalies and visualizations.

The project is unsupervised: there are no ground-truth anomaly labels. Therefore, detected anomalies should be interpreted as statistically unusual records, not automatically as errors or fraud.

## Dataset

The analysis uses NYC TLC Yellow Taxi trip data.

Expected input files:

```text
data/raw/yellow_tripdata_2026-03.parquet
data/taxi_zone_lookup.csv
```

The Parquet file contains individual taxi trips. The lookup CSV maps taxi zone IDs to borough names, which are used to interpret pickup and dropoff locations.

If the raw trip file is not included in the repository, place it manually in:

```text
data/raw/
```

## Main Features

The first preprocessing stage creates a basic feature matrix `X_scaled` with 22 features, including:

- passenger count,
- trip distance,
- trip duration,
- fare amount,
- tip amount,
- tolls amount,
- airport fee,
- pickup hour,
- pickup day of week,
- pickup and dropoff borough indicators.

Then the notebook creates an extended feature matrix `X_scaled_ext` with 38 features. Additional engineered features include:

- `speed_mph` — trip speed in miles per hour,
- `fare_per_mile` — fare amount divided by trip distance,
- `fare_per_minute` — fare amount divided by duration,
- `tip_rate` — tip amount relative to fare amount,
- `total_to_fare_ratio` — total amount relative to fare amount,
- `has_toll` — whether the trip includes tolls,
- `has_airport_fee` — whether the trip includes an airport fee,
- `same_borough` — whether pickup and dropoff are in the same borough,
- cyclic time features for pickup hour and pickup day of week,
- weekend and rush-hour indicators.

These extended features are important because an unusual trip is not necessarily just the longest or most expensive one. Often, the unusual part is the relation between variables, for example a very high fare per mile or fare per minute.

## Methods

### PCA

PCA is used to visualize the global linear structure of the data. It helps show whether trips differ mainly by distance, cost, route type or financial ratios.

The notebook also analyzes PCA loadings to understand which original features influence the first two principal components.

### UMAP

UMAP is used to visualize the local structure of the data in two dimensions. It helps reveal groups of similar trips, such as typical short Manhattan trips, airport-related trips or inter-borough trips.

UMAP does not create clusters by itself. It only determines the 2D position of points. Colors on UMAP plots come from KMeans labels, anomaly scores or selected feature values.

### KMeans Clustering

KMeans is used to divide trips into five clusters. The clusters are interpreted by comparing their profiles: average distance, duration, cost, airport fee share, toll share and dominant pickup/dropoff boroughs.

In the current run, the main cluster profiles are:

- cluster 0: typical short Manhattan → Manhattan trips,
- cluster 3: longer airport-related trips, mostly Queens → Manhattan,
- cluster 4: longer inter-borough trips, mostly Manhattan → Brooklyn,
- clusters 1 and 2: very small one-record clusters, interpreted as possible outliers rather than typical trip groups.

### Anomaly Detection

The project combines three anomaly-related signals:

- Isolation Forest score,
- Local Outlier Factor score,
- PCA reconstruction error.

These scores are normalized and averaged into:

```text
consensus_anomaly_score
```

This score is used as an exploratory measure of how unusual a trip is compared with the analyzed sample.

## Key Findings

The analysis shows that most trips form a large group of typical short urban rides, especially within Manhattan.

Additional engineered features make the analysis more informative. Features such as `fare_per_mile`, `fare_per_minute`, `has_airport_fee` and `same_borough` help distinguish between:

- ordinary short city trips,
- airport trips,
- longer inter-borough routes,
- statistically unusual records.

The most extreme anomaly found in the analyzed sample is a very short trip with a very high fare, resulting in an extremely high `fare_per_mile`. 

## Generated Outputs

The notebook generates interactive HTML visualizations in:

```text
plots/
```

Important plots include:

```text
plots/extension_pca_trip_distance.html
plots/extension_pca_total_amount.html
plots/extension_pca_fare_per_mile.html
plots/extension_pca_speed_mph.html
plots/extension_pca_loadings.html
plots/extension_umap_clusters.html
plots/extension_umap_anomaly_score.html
plots/extension_umap_fare_per_mile.html
plots/extension_umap_airport_fee.html
plots/extension_umap_same_borough.html
plots/extension_distance_cost_anomaly.html
plots/extension_anomaly_score_by_cluster.html
plots/extension_anomaly_score_histogram.html
```

The notebook also saves result tables in:

```text
results/
```

Important result files include:

```text
results/extension_cluster_profiles.csv
results/extension_top_anomalies.csv
results/extension_top_anomalies_explained.csv
```

## Project Structure

```text
NYC-Taxi-Anomaly-Detection/
│
├── data/
│   ├── raw/
│   │   └── yellow_tripdata_2026-03.parquet
│   └── taxi_zone_lookup.csv
│
├── notebooks/
│   └── start.ipynb
│
├── plots/
│   └── generated interactive HTML plots
│
├── results/
│   └── generated CSV result tables
│
├── utils/
│   └── preprocessing/
│       └── preprocess_yellow.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Quick Start

1. **Clone the repository**:
```bash
  git clone https://github.com/vvovvel/NYC-Taxi-Anomaly-Detection.git

```

2. **Install dependencies**:

```bash
    pip install -r requirements.txt
  ```
    
3.  **Run the analysis**: Explore the notebook located in the `notebooks/` directory.
