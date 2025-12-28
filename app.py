# app.py - Streamlit Book Recommender (Fixed for PyTorch 2.6+)

import streamlit as st
import torch
import pandas as pd
import pickle
import os
from torch_geometric.utils import degree

# Critical fix for PyTorch 2.6+ compatibility
from torch_geometric.data.data import DataTensorAttr, DataEdgeAttr
from torch_geometric.data.storage import GlobalStorage
import torch.serialization

torch.serialization.add_safe_globals([DataTensorAttr, DataEdgeAttr, GlobalStorage])

st.set_page_config(page_title="LightGCN Book Recommender", layout="centered")
st.title("📚 Graph-Based Book Recommendation System")
st.markdown("Built with **LightGCN** on the Book-Crossing dataset (filtered subset)")

@st.cache_resource
def load_all_data():
    st.info("Loading model and data... (first time may take 10-20 seconds)")

    # Load mappings
    with open('data/processed/user_mapping.pkl', 'rb') as f:
        user_map = pickle.load(f)
    user_to_node = user_map['user_to_node']

    with open('data/processed/book_mapping.pkl', 'rb') as f:
        book_map = pickle.load(f)
    node_to_book = book_map['node_to_book']

    book_metadata = pd.read_csv('data/processed/book_metadata.csv').set_index('ISBN')
    ratings = pd.read_csv('data/processed/filtered_ratings.csv')

    # Load graph and edges (with safe globals already added above)
    data = torch.load('data/processed/graph_data.pt', map_location='cpu')
    num_users = data.num_users
    num_books = data.num_books
    train_edge_index = torch.load('data/processed/train_edge_index.pt', map_location='cpu')

    # Model
    class LightGCN(torch.nn.Module):
        def __init__(self, num_users, num_books, embedding_dim=64, num_layers=3):
            super().__init__()
            self.num_users = num_users
            self.num_books = num_books
            self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
            self.item_embedding = torch.nn.Embedding(num_books, embedding_dim)
            torch.nn.init.normal_(self.user_embedding.weight, std=0.01)
            torch.nn.init.normal_(self.item_embedding.weight, std=0.01)

        def forward(self, edge_index):
            x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            outs = [x]
            row, col = edge_index
            deg = degree(row, num_nodes=x.size(0))
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            sparse = torch.sparse_coo_tensor(edge_index, norm, (x.size(0), x.size(0)))
            for _ in range(3):
                x = sparse @ x
                outs.append(x)
            final = sum(outs) / len(outs)
            return torch.split(final, [self.num_users, self.num_books])

    model = LightGCN(num_users, num_books)
    model_path = 'models/best_lightgcn.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        st.success("✅ Trained LightGCN model loaded!")
    else:
        st.warning("⚠️ No trained model found — using random embeddings")

    return {
        'user_to_node': user_to_node,
        'node_to_book': node_to_book,
        'book_metadata': book_metadata,
        'ratings': ratings,
        'model': model,
        'train_edge_index': train_edge_index,
        'num_books': num_books
    }

# Load data
data_dict = load_all_data()

# Recommendation function
@st.cache_data
def recommend(user_id: int, k: int = 10):
    if user_id not in data_dict['user_to_node']:
        return None

    model = data_dict['model']
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(data_dict['train_edge_index'])

    u_node = data_dict['user_to_node'][user_id]
    scores = user_emb[u_node] @ item_emb.t()

    user_rated = set(data_dict['ratings'][data_dict['ratings']['User-ID'] == user_id]['ISBN'].tolist())
    mask = torch.tensor([data_dict['node_to_book'][i] in user_rated for i in range(data_dict['num_books'])])
    scores[mask] = -float('inf')

    _, top_idx = torch.topk(scores, k + len(user_rated))
    top_isbns = []
    for idx in top_idx:
        isbn = data_dict['node_to_book'][idx.item()]
        if isbn not in user_rated:
            top_isbns.append(isbn)
        if len(top_isbns) == k:
            break

    if not top_isbns:
        return pd.DataFrame()

    rec_df = pd.DataFrame({'ISBN': top_isbns})
    rec_df = rec_df.merge(data_dict['book_metadata'][['Book-Title', 'Book-Author']].reset_index(),
                          on='ISBN', how='left')
    rec_df['Book-Title'] = rec_df['Book-Title'].fillna('Unknown Title')
    rec_df['Book-Author'] = rec_df['Book-Author'].fillna('Unknown Author')
    rec_df = rec_df[['Book-Title', 'Book-Author']]
    rec_df.columns = ['Title', 'Author']
    return rec_df

# UI
st.sidebar.header("🔍 Find Recommendations")
user_input = st.sidebar.text_input("Enter User-ID (e.g., 242, 254, 507)", value="242")

if st.sidebar.button("Get Recommendations", type="primary"):
    try:
        user_id = int(user_input)
        with st.spinner("Thinking..."):
            recs = recommend(user_id)
        if recs is None:
            st.error(f"User {user_id} not found in dataset")
        elif recs.empty:
            st.warning("No new recommendations (user rated everything!)")
        else:
            st.success(f"Top {len(recs)} recommendations for User {user_id}")
            st.dataframe(recs, use_container_width=True)
    except ValueError:
        st.error("Please enter a valid number")

# Info
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown(f"• {len(data_dict['user_to_node'])} users")
st.sidebar.markdown(f"• {data_dict['num_books']} books")
st.sidebar.markdown(f"• ~47k ratings")

