# E-commerce Product Recommendation System


```python

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class DataProcessor:
    def __init__(self):
        self.user_mapping = {}
        self.item_mapping = {}
        
    def load_sample_data(self):
        """Create sample e-commerce data"""
        
        n_users = 1000
        n_items = 500
        n_interactions = 10000
        
        users = np.random.randint(0, n_users, n_interactions)
        items = np.random.randint(0, n_items, n_interactions)
        ratings = np.random.randint(1, 6, n_interactions)
        
        df = pd.DataFrame({'user_id': users, 'item_id': items, 'rating': ratings})
        return df
    
    def create_interaction_matrix(self, df, user_col='user_id', item_col='item_id', value_col='rating'):
        """Create user-item interaction matrix"""
        # Create mappings
        unique_users = df[user_col].unique()
        unique_items = df[item_col].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        
        # Create matrix
        user_indices = [self.user_mapping[user] for user in df[user_col]]
        item_indices = [self.item_mapping[item] for item in df[item_col]]
        
        interaction_matrix = csr_matrix(
            (df[value_col], (user_indices, item_indices)),
            shape=(len(unique_users), len(unique_items))
        )
        
        return interaction_matrix
```


```python

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

class CollaborativeFiltering:
    def __init__(self):
        self.user_similarity = None
        self.item_similarity = None
        
    def user_based_cf(self, interaction_matrix, k=10):
        """User-based collaborative filtering"""
        
        self.user_similarity = cosine_similarity(interaction_matrix)
        
       
        similar_users = []
        for i in range(self.user_similarity.shape[0]):
            similar_indices = np.argsort(self.user_similarity[i])[::-1][1:k+1]
            similar_users.append(similar_indices)
            
        return similar_users
    
    def item_based_cf(self, interaction_matrix, k=10):
        """Item-based collaborative filtering"""
        # Calculate item similarity
        self.item_similarity = cosine_similarity(interaction_matrix.T)
        
        
        similar_items = []
        for i in range(self.item_similarity.shape[0]):
            similar_indices = np.argsort(self.item_similarity[i])[::-1][1:k+1]
            similar_items.append(similar_indices)
            
        return similar_items
    
    def predict_ratings(self, interaction_matrix, user_based=True):
        """Predict ratings using collaborative filtering"""
        if user_based:
            if self.user_similarity is None:
                self.user_based_cf(interaction_matrix)
            
            
            user_means = interaction_matrix.mean(axis=1)
            interaction_matrix_centered = interaction_matrix - user_means.reshape(-1, 1)
            
           
            pred = user_means.reshape(-1, 1) + self.user_similarity.dot(interaction_matrix_centered) / np.array([np.abs(self.user_similarity).sum(axis=1)]).T
            
        else:
            if self.item_similarity is None:
                self.item_based_cf(interaction_matrix)
            
            
            item_means = interaction_matrix.mean(axis=0)
            interaction_matrix_centered = interaction_matrix - item_means.reshape(1, -1)
            
           
            pred = item_means.reshape(1, -1) + interaction_matrix_centered.dot(self.item_similarity) / np.array([np.abs(self.item_similarity).sum(axis=1)])
        
        return pred
```


```python

import numpy as np
from scipy.sparse.linalg import svds

class MatrixFactorization:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, interaction_matrix):
        """Perform SVD matrix factorization"""
       
        user_means = np.array(interaction_matrix.mean(axis=1)).flatten()
        interaction_matrix_centered = interaction_matrix - user_means.reshape(-1, 1)
        
       
        U, sigma, Vt = svds(interaction_matrix_centered, k=self.n_factors)
        
        
        sigma = np.diag(sigma)
        self.user_factors = U
        self.item_factors = Vt.T
        
        return U.dot(sigma).dot(Vt) + user_means.reshape(-1, 1)
    
    def predict(self, user_idx, item_idx):
        """Predict rating for a user-item pair"""
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.user_factors[user_idx, :].dot(self.item_factors[item_idx, :].T)
```


```python

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dot

class NeuralCF:
    def __init__(self, n_users, n_items, embedding_dim=50):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
        
    def _build_model(self):
        """Build neural collaborative filtering model"""
        # User input
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(self.n_users, self.embedding_dim, name='user_embedding')(user_input)
        user_vec = Flatten()(user_embedding)
        
        # Item input
        item_input = Input(shape=(1,), name='item_input')
        item_embedding = Embedding(self.n_items, self.embedding_dim, name='item_embedding')(item_input)
        item_vec = Flatten()(item_embedding)
        
        # Merge layers
        concat = Concatenate()([user_vec, item_vec])
        
        # Add fully connected layers
        dense1 = Dense(64, activation='relu')(concat)
        dense2 = Dense(32, activation='relu')(dense1)
        dense3 = Dense(16, activation='relu')(dense2)
        
        # Output layer
        output = Dense(1, activation='linear')(dense3)
        
        # Build model
        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, user_ids, item_ids, ratings, epochs=10, batch_size=64, validation_split=0.1):
        """Train the model"""
        history = self.model.fit(
            [user_ids, item_ids], ratings,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        return history
    
    def predict(self, user_ids, item_ids):
        """Make predictions"""
        return self.model.predict([user_ids, item_ids])
```


```python

import networkx as nx
import numpy as np

class GraphRecommender:
    def __init__(self):
        self.graph = nx.Graph()
        
    def build_user_item_graph(self, df, user_col='user_id', item_col='item_id'):
        """Build bipartite graph of users and items"""
        
        for user in df[user_col].unique():
            self.graph.add_node(user, type='user')
            
        for item in df[item_col].unique():
            self.graph.add_node(item, type='item')
        
       
        for _, row in df.iterrows():
            self.graph.add_edge(row[user_col], row[item_col], weight=row.get('rating', 1))
            
        return self.graph
    
    def recommend_items_personalized_pagerank(self, user_id, top_n=10):
        """Recommend items using personalized PageRank"""
       
        personalization = {node: 0 for node in self.graph.nodes()}
        personalization[user_id] = 1
        
        # Calculate personalized PageRank
        ppr = nx.pagerank(self.graph, personalization=personalization)
        
        # Get top items (excluding users)
        item_scores = {
            node: score for node, score in ppr.items() 
            if self.graph.nodes[node].get('type') == 'item'
        }
        
        
        recommended_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return recommended_items
    
    def find_similar_users(self, user_id, top_n=5):
        """Find similar users using graph metrics"""
        
        user_items = set(self.graph.neighbors(user_id))
        
        similar_users = []
        for other_user in self.graph.nodes():
            if self.graph.nodes[other_user].get('type') == 'user' and other_user != user_id:
                other_items = set(self.graph.neighbors(other_user))
                
                
                intersection = len(user_items.intersection(other_items))
                union = len(user_items.union(other_items))
                similarity = intersection / union if union > 0 else 0
                
                similar_users.append((other_user, similarity))
        
        
        return sorted(similar_users, key=lambda x: x[1], reverse=True)[:top_n]
```


```python

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os

from src.data_processing import DataProcessor
from src.collaborative_filtering import CollaborativeFiltering
from src.matrix_factorization import MatrixFactorization
from src.neural_cf import NeuralCF
from src.graph_recommendations import GraphRecommender

app = FastAPI(title="E-commerce Recommendation API")


data_processor = DataProcessor()
cf_model = CollaborativeFiltering()
mf_model = MatrixFactorization()
graph_model = GraphRecommender()


df = data_processor.load_sample_data()
interaction_matrix = data_processor.create_interaction_matrix(df)


cf_model.user_based_cf(interaction_matrix)
mf_model.fit(interaction_matrix)
graph_model.build_user_item_graph(df)


n_users = len(data_processor.user_mapping)
n_items = len(data_processor.item_mapping)
neural_cf = NeuralCF(n_users, n_items)

class RecommendationRequest(BaseModel):
    user_id: int
    method: str = "collaborative"  # collaborative, matrix, neural, graph
    top_n: int = 10

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    try:
        user_idx = data_processor.user_mapping.get(request.user_id)
        if user_idx is None:
            return {"message": "User not found", "recommendations": []}
        
        if request.method == "collaborative":
            # Get user-based predictions
            predictions = cf_model.predict_ratings(interaction_matrix, user_based=True)
            user_predictions = predictions[user_idx, :].toarray().flatten()
            
            # Get top N items
            top_items_idx = np.argsort(user_predictions)[::-1][:request.top_n]
            recommendations = [
                {"item_id": list(data_processor.item_mapping.keys())[list(data_processor.item_mapping.values()).index(idx)], 
                 "score": float(user_predictions[idx])}
                for idx in top_items_idx
            ]
            
        elif request.method == "matrix":
            # Use matrix factorization
            item_scores = []
            for item_id, item_idx in data_processor.item_mapping.items():
                score = mf_model.predict(user_idx, item_idx)
                item_scores.append((item_id, score))
            
            # Sort and get top N
            item_scores.sort(key=lambda x: x[1], reverse=True)
            recommendations = [{"item_id": item_id, "score": float(score)} 
                              for item_id, score in item_scores[:request.top_n]]
            
        elif request.method == "graph":
            # Use graph-based recommendations
            graph_recommendations = graph_model.recommend_items_personalized_pagerank(
                request.user_id, request.top_n
            )
            recommendations = [{"item_id": item_id, "score": float(score)} 
                              for item_id, score in graph_recommendations]
            
        elif request.method == "neural":
            # This would need proper implementation with trained model
            return {"message": "Neural CF not fully implemented in this demo"}
            
        else:
            raise HTTPException(status_code=400, detail="Invalid method specified")
        
        return {"user_id": request.user_id, "method": request.method, "recommendations": recommendations}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "E-commerce Recommendation API", "status": "active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[11], line 7
          4 import pickle
          5 import os
    ----> 7 from src.data_processing import DataProcessor
          8 from src.collaborative_filtering import CollaborativeFiltering
          9 from src.matrix_factorization import MatrixFactorization
    

    ModuleNotFoundError: No module named 'src'



```python
pip install fastapi uvicorn pandas numpy scipy scikit-learn tensorflow networkx pydantic
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: fastapi in c:\users\harikesh\appdata\roaming\python\python312\site-packages (0.116.1)
    Requirement already satisfied: uvicorn in c:\users\harikesh\appdata\roaming\python\python312\site-packages (0.35.0)
    Requirement already satisfied: pandas in c:\programdata\anaconda3\lib\site-packages (2.2.2)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (1.26.4)
    Requirement already satisfied: scipy in c:\programdata\anaconda3\lib\site-packages (1.13.1)
    Requirement already satisfied: scikit-learn in c:\programdata\anaconda3\lib\site-packages (1.5.1)
    Requirement already satisfied: tensorflow in c:\users\harikesh\appdata\roaming\python\python312\site-packages (2.19.0)
    Requirement already satisfied: networkx in c:\programdata\anaconda3\lib\site-packages (3.3)
    Requirement already satisfied: pydantic in c:\programdata\anaconda3\lib\site-packages (2.8.2)
    Requirement already satisfied: starlette<0.48.0,>=0.40.0 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from fastapi) (0.47.3)
    Requirement already satisfied: typing-extensions>=4.8.0 in c:\programdata\anaconda3\lib\site-packages (from fastapi) (4.11.0)
    Requirement already satisfied: click>=7.0 in c:\programdata\anaconda3\lib\site-packages (from uvicorn) (8.1.7)
    Requirement already satisfied: h11>=0.8 in c:\programdata\anaconda3\lib\site-packages (from uvicorn) (0.14.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\programdata\anaconda3\lib\site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in c:\programdata\anaconda3\lib\site-packages (from pandas) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in c:\programdata\anaconda3\lib\site-packages (from pandas) (2023.3)
    Requirement already satisfied: joblib>=1.2.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (3.5.0)
    Requirement already satisfied: absl-py>=1.0.0 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (2.3.1)
    Requirement already satisfied: astunparse>=1.6.0 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (1.6.3)
    Requirement already satisfied: flatbuffers>=24.3.25 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (25.2.10)
    Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (0.6.0)
    Requirement already satisfied: google-pasta>=0.1.1 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: libclang>=13.0.0 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (18.1.1)
    Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (3.4.0)
    Requirement already satisfied: packaging in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (24.1)
    Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (4.25.3)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (2.32.3)
    Requirement already satisfied: setuptools in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (75.1.0)
    Requirement already satisfied: six>=1.12.0 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (1.16.0)
    Requirement already satisfied: termcolor>=1.1.0 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (3.1.0)
    Requirement already satisfied: wrapt>=1.11.0 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (1.14.1)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (1.74.0)
    Requirement already satisfied: tensorboard~=2.19.0 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (2.19.0)
    Requirement already satisfied: keras>=3.5.0 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (3.10.0)
    Requirement already satisfied: h5py>=3.11.0 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (3.11.0)
    Requirement already satisfied: ml-dtypes<1.0.0,>=0.5.1 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorflow) (0.5.1)
    Requirement already satisfied: annotated-types>=0.4.0 in c:\programdata\anaconda3\lib\site-packages (from pydantic) (0.6.0)
    Requirement already satisfied: pydantic-core==2.20.1 in c:\programdata\anaconda3\lib\site-packages (from pydantic) (2.20.1)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\programdata\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow) (0.44.0)
    Requirement already satisfied: colorama in c:\programdata\anaconda3\lib\site-packages (from click>=7.0->uvicorn) (0.4.6)
    Requirement already satisfied: rich in c:\programdata\anaconda3\lib\site-packages (from keras>=3.5.0->tensorflow) (13.7.1)
    Requirement already satisfied: namex in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from keras>=3.5.0->tensorflow) (0.1.0)
    Requirement already satisfied: optree in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from keras>=3.5.0->tensorflow) (0.17.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\programdata\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorflow) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in c:\programdata\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorflow) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\programdata\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorflow) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in c:\programdata\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorflow) (2024.8.30)
    Requirement already satisfied: anyio<5,>=3.6.2 in c:\programdata\anaconda3\lib\site-packages (from starlette<0.48.0,>=0.40.0->fastapi) (4.2.0)
    Requirement already satisfied: markdown>=2.6.8 in c:\programdata\anaconda3\lib\site-packages (from tensorboard~=2.19.0->tensorflow) (3.4.1)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\users\harikesh\appdata\roaming\python\python312\site-packages (from tensorboard~=2.19.0->tensorflow) (0.7.2)
    Requirement already satisfied: werkzeug>=1.0.1 in c:\programdata\anaconda3\lib\site-packages (from tensorboard~=2.19.0->tensorflow) (3.0.3)
    Requirement already satisfied: sniffio>=1.1 in c:\programdata\anaconda3\lib\site-packages (from anyio<5,>=3.6.2->starlette<0.48.0,>=0.40.0->fastapi) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.1.1 in c:\programdata\anaconda3\lib\site-packages (from werkzeug>=1.0.1->tensorboard~=2.19.0->tensorflow) (2.1.3)
    Requirement already satisfied: markdown-it-py>=2.2.0 in c:\programdata\anaconda3\lib\site-packages (from rich->keras>=3.5.0->tensorflow) (2.2.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\programdata\anaconda3\lib\site-packages (from rich->keras>=3.5.0->tensorflow) (2.15.1)
    Requirement already satisfied: mdurl~=0.1 in c:\programdata\anaconda3\lib\site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.5.0->tensorflow) (0.1.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install -r requirements.txt
```

    Defaulting to user installation because normal site-packages is not writeable
    Note: you may need to restart the kernel to use updated packages.
    

    ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'
    


```python

```


```python

import random
import math
from collections import defaultdict
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

class SimpleRecommender:
    def __init__(self):
        self.user_items = defaultdict(dict)
        self.item_users = defaultdict(dict)
        self.user_similarities = {}
        
    def add_rating(self, user_id, item_id, rating):
        self.user_items[user_id][item_id] = rating
        self.item_users[item_id][user_id] = rating
        
    def generate_sample_data(self, n_users=100, n_items=50, n_ratings=500):
        for _ in range(n_ratings):
            user_id = random.randint(1, n_users)
            item_id = random.randint(1, n_items)
            rating = random.randint(1, 5)
            self.add_rating(user_id, item_id, rating)
    
    def cosine_similarity(self, vec1, vec2):
        common_keys = set(vec1.keys()) & set(vec2.keys())
        if not common_keys:
            return 0
            
        dot_product = sum(vec1[key] * vec2[key] for key in common_keys)
        norm1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        norm2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (norm1 * norm2)
    
    def calculate_user_similarities(self):
        user_ids = list(self.user_items.keys())
        self.user_similarities = {}
        
        for i, user1 in enumerate(user_ids):
            for user2 in user_ids[i+1:]:
                similarity = self.cosine_similarity(
                    self.user_items[user1], 
                    self.user_items[user2]
                )
                if user1 not in self.user_similarities:
                    self.user_similarities[user1] = {}
                self.user_similarities[user1][user2] = similarity
                
                if user2 not in self.user_similarities:
                    self.user_similarities[user2] = {}
                self.user_similarities[user2][user1] = similarity
    
    def recommend_items(self, user_id, top_n=5):
        if user_id not in self.user_items:
            return {"error": "User not found"}
            
       
        user_items = set(self.user_items[user_id].keys())
        recommendations = {}
        
        if user_id in self.user_similarities:
            similar_users = sorted(
                self.user_similarities[user_id].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        else:
            similar_users = []
            
        for similar_user, similarity in similar_users:
            for item_id, rating in self.user_items[similar_user].items():
                if item_id not in user_items:
                    if item_id not in recommendations:
                        recommendations[item_id] = 0
                    recommendations[item_id] += similarity * rating
        
       
        if not recommendations:
            popular_items = sorted(
                self.item_users.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:top_n]
            return [{"item_id": item_id, "score": len(users), "type": "popular"} 
                   for item_id, users in popular_items]
        
        sorted_recommendations = sorted(
            recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        return [{"item_id": item_id, "score": score, "type": "collaborative"} 
                for item_id, score in sorted_recommendations]

class RecommendationHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.recommender = SimpleRecommender()
        self.recommender.generate_sample_data()
        self.recommender.calculate_user_similarities()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path.startswith('/recommend/'):
            try:
                
                user_id = int(self.path.split('/')[-1])
                recommendations = self.recommender.recommend_items(user_id)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "user_id": user_id,
                    "recommendations": recommendations
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except ValueError:
                self.send_error(400, "Invalid user ID")
            except Exception as e:
                self.send_error(500, str(e))
        
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <html>
            <body>
                <h1>E-commerce Recommendation System</h1>
                <p>Use /recommend/&lt;user_id&gt; to get recommendations</p>
                <p>Example: <a href="/recommend/5">/recommend/5</a></p>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_POST(self):
        if self.path == '/recommend':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                user_id = data.get('user_id', 1)
                top_n = data.get('top_n', 5)
                
                recommendations = self.recommender.recommend_items(user_id, top_n)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "user_id": user_id,
                    "recommendations": recommendations
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                self.send_error(400, str(e))
        
        else:
            self.send_error(404, "Endpoint not found")

def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RecommendationHandler)
    print('Starting recommendation server on http://localhost:8000')
    print('Visit http://localhost:8000 in your browser')
    print('Use GET /recommend/<user_id> or POST /recommend with JSON body')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
```

    Starting recommendation server on http://localhost:8000
    Visit http://localhost:8000 in your browser
    Use GET /recommend/<user_id> or POST /recommend with JSON body
    
