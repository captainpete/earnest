import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from streamlit import rerun
from streamlit_searchbox import st_searchbox
from rapidfuzz import process, fuzz

names_file = 'baby_names_full.pkl'
labels_file = 'baby_labels.pkl'
model_file = 'baby_model.json'
n = 7

@st.cache_data
def load_data():
    return pd.read_pickle(names_file)

def labels_to_df(labels):
    return pd.DataFrame(labels, columns={'name': str, 'label': int, 'qid': int})

def load_labels():
    # check if file exists, or return empty data frame
    if Path(labels_file).is_file():
        return pd.read_pickle(labels_file)
    else:
        return labels_to_df([])

data = load_data()
labels = load_labels()

def update_labels(new_labels):
    global labels
    labels = pd.concat((labels, new_labels), axis=0, ignore_index=True)
    labels.to_pickle(labels_file)
    train_model()

def labels_from_selection(name, options):
    print(f"{name}: {', '.join(options)}")
    assert name in options
    max_qid = labels['qid'].max()
    next_qid = 0 if np.isnan(max_qid) else max_qid + 1
    return labels_to_df([(option, option == name, next_qid) for option in options])

def train_model():
    X_train = data.loc[labels['name']]
    y_train = labels['label']
    qid_train = labels['qid']

    ranker = xgb.XGBRanker(
        n_estimators=1000,
        tree_method="hist",
        device="cuda",
        learning_rate=0.01,
        subsample=1.0,
        colsample_bynode=0.2,
        sampling_method="gradient_based",
        objective="rank:ndcg",
        lambdarank_unbiased=True,
        lambdarank_bias_norm=1,
        lambdarank_num_pair_per_sample=12,
        lambdarank_pair_method="topk",
        ndcg_exp_gain=True,
    )
    ranker.fit(
        X_train,
        y_train,
        qid=qid_train,
        verbose=True,
    )
    ranker.save_model(model_file)
    return None

def load_model():
    if Path(model_file).is_file():
        ranker = xgb.XGBRanker()
        ranker.load_model(model_file)
        return ranker
    else:
        return None

ranker = load_model()

def sample_names():
    if ranker is None:
        return np.random.choice(data.index, n*3)

    X_pred = data.loc[data.index]
    scores = ranker.predict(X_pred)
    order = np.argsort(scores)[::-1]

    best = np.random.choice(data.index[order][:200], n, replace=False)
    good = np.random.choice(data.index[order][200:1000], n, replace=False)
    random = np.random.choice(data.index, n)
    return list(best) + list(good) + list(random)

names = sample_names()

def handle_click(key):
    name = names[key]
    new_labels = labels_from_selection(name, names)
    update_labels(new_labels)

for i, col in enumerate(st.columns(3)):
    for j in range(n):
        key = i * n + j
        col.button(
            names[key],
            key=key,
            on_click=handle_click,
            args=[key],
            use_container_width=True,
        )

def search_names(term: str) -> list:
    global data
    names = list(data.index)
    results = process.extract(term, names, limit=5, scorer=fuzz.QRatio)
    return [r[0] for r in results]

def select_custom_name(name: str):
    global names
    if name is None:
        return
    options = names if name in names else np.concatenate((names, [name]))
    new_labels = labels_from_selection(name, options)
    update_labels(new_labels)
    st.rerun()

@st.fragment
def search_box():
    st_searchbox(
        search_names,
        placeholder="Search names... ",
        key="search",
        # clear_on_submit=True,
        rerun_scope="fragment",
        submit_function=select_custom_name,
    )

search_box()

if st.button("🌷🌸🌹🌺🌻🌼", key='refresh', use_container_width=True):
    st.rerun()

st.markdown(f"{len(labels)} preferences")

col_best, col_worst = st.columns(2)
if ranker is not None:
    X_pred = data.loc[data.index]
    scores = ranker.predict(X_pred)
    order = np.argsort(scores)[::-1]

    k = 50
    top_k = pd.DataFrame({
        'name': data.index[order][:k],
        'score': scores[order][:k],
    })
    bottom_k = pd.DataFrame({
        'name': data.index[order][-k:],
        'score': scores[order][-k:],
    })

    col_best.subheader("the best")
    col_best.dataframe(top_k, use_container_width=True)

    col_worst.subheader("the worst")
    col_worst.dataframe(bottom_k, use_container_width=True)
