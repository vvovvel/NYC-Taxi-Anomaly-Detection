
# NYC Taxi Typology & Anomaly Detection

This project analyzes NYC TLC datasets to identify standard trip patterns and detect traffic anomalies using unsupervised learning and dimensionality reduction.

## Objectives

* **Feature Engineering**: Processing spatial, temporal, and financial trip data.
* **Dimensionality Reduction**: Comparative analysis of **PCA**, **t-SNE**, and **UMAP** for 2D data projection.
* **Pattern Discovery**: Unsupervised clustering to define trip typologies.
* **Anomaly Detection**: Identification of statistical outliers in urban mobility.
* **Interactive Viz**: Visualizing data topology using **Altair**.

---

## Project Structure

The repository is organized to maintain a clear separation between data, logic, and results:

* **`data/raw/`**: Storage for the initial Parquet datasets.
* **`notebooks/`**: Jupyter notebooks for exploratory data analysis.
* **`plots/`**: Output directory for generated interactive `.html` visualizations.
* **`utils/`**: Modularized Python scripts:
* `preprocessing/`: Data cleaning, sampling, and scaling.
* `visualization.py`: Shared plotting functions using Altair.


* **`main.py`**: The entry point script that executes the full processing and visualization pipeline.

---

## Tech Stack

* **Language**: Python 3.x
* **Analysis**: Pandas, Scikit-learn, UMAP-learn
* **Visualization**: Altair

---

## Setup & Data Acquisition

Follow these steps to get the project running locally:

1. **Clone the repository**:
```bash
  git clone https://github.com/your-username/taxi-anomaly-detection.git
  cd NYC-Taxi-Anomaly-Detection
```


2. **Install dependencies**:
```bash
  pip install -r requirements.txt
```


3. **Acquire the dataset**:
The analysis requires the March 2026 Yellow Taxi trip record.
   * **Download**: [NYC TLC Yellow Taxi - March 2026 (Parquet)](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-03.parquet)
   * **Placement**: Save the downloaded file into the `data/raw/` directory.


4. **Run the analysis**:
Execute the main pipeline to generate visualizations:
```bash
  python main.py
```
5. **Open HTMLs**, saved in `plots/` directory. 

Alternatively, explore the experimentation process in `notebooks/start.ipynb`.

---