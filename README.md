#  Customer Segmentation using Unsupervised Learning (KMeans)

This project applies **Exploratory Data Analysis (EDA)** and **Machine Learning (KMeans Clustering)** to segment mall customers based on spending habits, income, age, and gender. It provides insights for targeted marketing strategies.

---

##  Objective

- Segment customers into distinct clusters based on:
  - Age
  - Annual Income
  - Spending Score
  - Gender
- Visualize the clusters using PCA and t-SNE
- Provide data-driven insights for business decision-making

---

##  Approach

1. **EDA Class**
   - Load and clean the dataset (missing & duplicate removal)
   - Rename columns and format the data
   - Analyze:
     - Spending habits by gender
     - Average income and score by gender
   - Visualize:
     - Pie chart (Spending by Gender)
     - Bar chart (Income & Spending)

2. **ML Class**
   - Inherits EDA
   - One-hot encode gender
   - Standardize features
   - Apply **KMeans Clustering** (`k=5`)
   - Visualize clusters using:
     - **PCA** (linear projection)
     - **t-SNE** (non-linear projection)
   - Summary statistics for each cluster

---

## Results & Findings

- The dataset was successfully clustered into 5 meaningful groups.
- Clusters showed distinct behaviors:
  - Some clusters had younger, high-spending customers.
  - Some had older, low-spending customers.
  - One cluster was balanced across gender, income, and score.
- **t-SNE** provided more separated and clearer visualizations than **PCA**.

---

##  How to Run

# Install dependencies
pip install -r requirements.txt

# Run the script
python mall_customer_segmentation.py
