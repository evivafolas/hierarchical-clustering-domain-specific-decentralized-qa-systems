import numpy as np
import matplotlib.pyplot as plt

data = {
    'sentence-transformers/gtr-t5-base': {
        'business': {'f1': 0.8137142857142857},
        'entertainment': {'f1': 0.9062500000000001},
        'politics': {'f1': 0.7111111111111112},
        'sport': {'f1': 0.9805825242718446},
        'tech': {'f1': 0.7816711590296496}
    },
    'sentence-transformers/all-mpnet-base-v2': {
        'business': {'f1': 0.8137142857142857},
        'entertainment': {'f1': 0.9062500000000001},
        'politics': {'f1': 0.7111111111111112},
        'sport': {'f1': 0.9805825242718446},
        'tech': {'f1': 0.7816711590296496}
    },
    'sentence-transformers/all-distilroberta-v1': {
        'business': {'f1': 0.8137142857142857},
        'entertainment': {'f1': 0.9062500000000001},
        'politics': {'f1': 0.7111111111111112},
        'sport': {'f1': 0.9805825242718446},
        'tech': {'f1': 0.7816711590296496}
    },
    'sentence-transformers/all-MiniLM-L6-v2': {
        'business': {'f1': 0.8137142857142857},
        'entertainment': {'f1': 0.9062500000000001},
        'politics': {'f1': 0.7111111111111112},
        'sport': {'f1': 0.9805825242718446},
        'tech': {'f1': 0.7816711590296496}
    },
    'sentence-transformers/all-MiniLM-L12-v2': {
        'business': {'f1': 0.8137142857142857},
        'entertainment': {'f1': 0.9062500000000001},
        'politics': {'f1': 0.7111111111111112},
        'sport': {'f1': 0.9805825242718446},
        'tech': {'f1': 0.7816711590296496}
    }
}

models = list(data.keys())
categories = list(data[models[0]].keys())
f1_scores = {category: [data[model][category]['f1'] for model in models] for category in categories}

n_models = len(models)
bar_width = 0.15
index = np.arange(n_models)

plt.figure(figsize=(12, 6))

for i, (category, scores) in enumerate(f1_scores.items()):
    plt.bar(index + i * bar_width, scores, bar_width, label=category)

plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('F1 Scores for Different Models and Categories')
plt.xticks(index + (n_models / 2) * bar_width, models, rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.show()
