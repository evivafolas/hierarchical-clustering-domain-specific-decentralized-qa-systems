import numpy as np
import matplotlib.pyplot as plt

data = {
    'MiniLM L12 v2': {
        'Accuracy Stock': 0.8543820224719101,
        'Accuracy w/ n-gram range = (1, 2) and n=15': 0.8660674157303371,
        'Accuracy w/ n-gram range = (1, 3) and n=15': 0.8741573033707866,
        'Accuracy w/ n-gram range = (1, 5) and n=15': 0.8561797752808988,
    },
    'MiniLM L6 v2': {
        'Accuracy Stock': 0.8624719101123596,
        'Accuracy w/ n-gram range = (1, 2) and n=15': 0.8382022471910112,
        'Accuracy w/ n-gram range = (1, 3) and n=15': 0.9011235955056179,
        'Accuracy w/ n-gram range = (1, 5) and n=15': 0.9456179775280898
    },
    'DistilRoBERTa v1': {
        'Accuracy Stock': 0.8355056179775281,
        'Accuracy w/ n-gram range = (1, 2) and n=15': 0.9056179775280899,
        'Accuracy w/ n-gram range = (1, 3) and n=15': 0.9141573033707865,
        'Accuracy w/ n-gram range = (1, 5) and n=15': 0.941123595505618
    },
    'MPNET Base v2': {
        'Accuracy Stock': 0.8386516853932584,
        'Accuracy w/ n-gram range = (1, 2) and n=15': 0.8629213483146068,
        'Accuracy w/ n-gram range = (1, 3) and n=15': 0.8665168539325843,
        'Accuracy w/ n-gram range = (1, 5) and n=15': 0.9159550561797752
    },
    'GTR T5 Base': {
        'Accuracy Stock': 0.7451685393258427,
        'Accuracy w/ n-gram range = (1, 2) and n=15': 0.8525842696629213,
        'Accuracy w/ n-gram range = (1, 3) and n=15': 0.8476404494382023,
        'Accuracy w/ n-gram range = (1, 5) and n=15': 0.8
    }
}

models = list(data.keys())
metrics = list(data[models[0]].keys())

# Create sublists for each metric
metric_values = {metric: [data[model][metric] for model in models] for metric in metrics}

n_models = len(models)
bar_width = 0.15
index = np.arange(n_models)

plt.figure(figsize=(12, 6))

for i, (metric, values) in enumerate(metric_values.items()):
    plt.bar(index + i * bar_width, values, bar_width, label=metric)

plt.xlabel('Models')
plt.ylabel('Document Classification Accuracy')
plt.title('Document Classification for Different Models')
plt.xticks(index + (n_models / 2) * bar_width, models, rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.show()
