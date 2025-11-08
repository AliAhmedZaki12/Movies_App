# Movies_App (https://moviesapp-aa56.streamlit.app/)

#  (Advanced Movie Clustering System)

##  (Project Description)
This project presents a complete and well-structured **unsupervised learning pipeline** for clustering complex datasets using both **KMeans** and **HDBSCAN**, with an additional **Meta-Clustering** layer to combine multiple clustering insights into a single unified model.

It was built to explore latent patterns in mixed-type datasets (numeric and categorical), automatically handle preprocessing, and visualize the resulting clusters in two-dimensional PCA space.  
The workflow follows professional machine learning engineering practices with clean modularization, interpretability, and reproducibility.

---

##  (Motivation)
Clustering is one of the most powerful yet underused tools in data science — particularly when the dataset contains both numerical and categorical features.  
This project provides a **universal, plug-and-play clustering framework** that:
- Automatically preprocesses heterogeneous data types.
- Tests multiple clustering strategies.
- Evaluates cluster quality using **Silhouette Score**.
- Combines multiple clustering results through **Meta-Clustering** for better stability and insight.

The example dataset (`movies.csv`) demonstrates the framework on real-world structured data.

---

##  (Key Features)
- **Dynamic Column Selection:** Users can specify which columns to include through `selected_cols`.  
- **Full Data Exploration:** Prints key dataset statistics, missing values, and distributions.  
- **Comprehensive EDA:**  
  - Histograms with KDE for numeric columns.  
  - Boxplots for outlier inspection.  
  - Countplots for categorical distributions.  
  - Correlation heatmap for numeric relationships.  
- **Automated Preprocessing:**  
  - `ColumnTransformer` handling both numeric and categorical pipelines.  
  - `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.  
  - Persistent preprocessing model saved via `joblib`.  
- **Dimensionality Reduction:**  
  - PCA (2D) for visualization and variance analysis.  
- **Dual Clustering Models:**  
  - **KMeans:** GridSearch over *k = 2 to 10*.  
  - **HDBSCAN:** Multiple configurations for `min_cluster_size` and `min_samples`.  
  - Evaluated using Silhouette Score.  
- **Meta-Clustering (Stacking):**  
  - Aggregates results from all models using KMeans to form robust meta-clusters.  
- **Rich Visualization:**  
  - Cluster scatterplots (Best Model & Meta-Clustering).  
- **Output:**  
  - Saves results with `BestCluster` and `MetaCluster` columns to `clustered_movies_output.csv`.

---

## 🧩 Project Structure

