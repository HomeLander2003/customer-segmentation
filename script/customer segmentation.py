#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# using logging to track events that happen when  program runs
logging.basicConfig(
    level=logging.INFO,                                  
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# class to perform EDA
class EDA:
    
    def __init__(self):
        self.file = None  # instance variable

    def clean(self):
        try:
            file_path = r"Mall_Customers.csv"# add file path according to your file location
            
            if os.path.isfile(file_path):
                self.file = pd.read_csv(file_path)
                print(self.file.head())
                print(self.file.info())
                print(np.shape(self.file))
                print(self.file.isnull().sum())
                print(self.file.duplicated().sum())
                
                self.file.rename(columns={"Spending Score (1-100)": "Spending_Score", "Genre": "Gender"}, inplace=True)



                # Remove rows with any missing values
                if self.file.isnull().values.any():         
                    self.file.dropna(inplace=True)

                # Remove duplicate rows
                if self.file.duplicated().any():             
                    self.file.drop_duplicates(keep="first", inplace=True)
                    self.file.reset_index(drop=True, inplace=True)
            
            else:
                logging.error("File path not correct or file not found.")
        
        except Exception as e:
            logging.error(f"Exception during file cleaning: {e}")
    
    def analysis(self):
        if self.file is None or self.file.empty:
            logging.error("File not found or is empty. Cannot perform analysis.")
        else:
            print(self.file.describe())
            
            gr1=self.file.groupby("Gender")["Spending_Score"].mean()
            print(gr1)
            
            gr2=self.file.groupby("Gender")[["Annual Income (k$)","Spending_Score"]].mean().reset_index()
            print(gr2)
            
            # Melt the DataFrame to long format for seaborn
            gr2_melted=gr2.melt(id_vars="Gender", 
            value_vars=["Annual Income (k$)", "Spending_Score"],
            var_name="Metric", 
            value_name="Value")
            
            return gr1,gr2_melted
            
    def visualize(self):
        
        def pie(gr1):
        
            plt.figure(figsize=(8, 8))
            plt.title("Comparing Spending Scores Btw Genders",color="purple",weight="bold",fontsize=20)
            plt.pie(gr1.values, labels=gr1.index, autopct="%.1f%%", startangle=90, wedgeprops={'width': 0.6})

            #  donut chart with white center
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)

            plt.tight_layout()
            plt.show()
            
        pie(gr1)
        
        
        def bar(gr2_melted):
            
            plt.figure(figsize=(8, 8))
            plt.title("Distribution of Income And Spending_Scores Btw Genders",color="purple",weight="bold",fontsize=20)
            sns.barplot(x="Gender",y="Value",hue="Metric",data=gr2_melted,palette="coolwarm")


            plt.tight_layout()
            plt.show()
            
        bar(gr2_melted)
        
class ML(EDA):
    
    def FE(self):
        
    #  the one‑hot‑encoded version
        self.file_encoded = pd.get_dummies(
            self.file,
            columns=["Gender"],
            drop_first=True,
            dtype=int
        )
        print(self.file_encoded.head())
        
    def ml(self, k=5):
        if not hasattr(self, "file_encoded"):
            logging.error("Run FE() first.")
            return

        # select features
        features = ["Age", "Annual Income (k$)", "Spending_Score", "Gender_Male"]
        X = self.file_encoded[features]

        # scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # cluster
        kmeans = KMeans(n_clusters=k, random_state=101)
        self.file_encoded["Cluster"] = kmeans.fit_predict(X_scaled)

        # quick sanity check
        print(self.file_encoded.groupby("Cluster")[features].mean())
        
    
    def plot_clusters(self, technique="pca", perplexity=30):
       
        if "Cluster" not in self.file_encoded.columns:
            logging.error("Run ml() first to generate clusters.")
            return

        features = ["Age", "Annual Income (k$)", "Spending_Score", "Gender_Male"]
        X = self.file_encoded[features].values

        # ----- scale again so visualisation uses same space -----
        X_scaled = StandardScaler().fit_transform(X)

        # ----- choose reducer -----
        if technique == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            title = "PCA"
        else:                       # t-SNE
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                init="pca",
                learning_rate="auto"
            )
            title = f"t‑SNE (perp={perplexity})"

        X_2d = reducer.fit_transform(X_scaled)

        # ----- quick scatter -----
        plt.figure(figsize=(7, 6))
        palette = sns.color_palette("hls", self.file_encoded["Cluster"].nunique())
        sns.scatterplot(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            hue=self.file_encoded["Cluster"],
            palette=palette,
            s=60,
            alpha=0.85
        )
        plt.title(f"K‑Means clusters visualised with {title}", weight="bold")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
# ------------------------------------------------------------------
        centroids = variable.file_encoded.groupby("Cluster")[["Age","Annual Income (k$)", "Spending_Score"]].mean()
        genders   = variable.file_encoded.groupby("Cluster")["Gender_Male"].mean()
        summary   = pd.concat([centroids, genders.rename("Pct_Male")], axis=1)
        print(summary.round(1))


       
   

# Create an instance of the class and call the methods
variable = ML()
variable.clean()
gr1,gr2_melted=variable.analysis()
# variable.visualize()
variable.FE()
variable.ml()
variable.plot_clusters("pca")          # linear view
variable.plot_clusters("tsne", 25)     # non‑linear view

