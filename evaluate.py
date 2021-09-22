from CustomToTangentSpace import CustomToTangentSpace
import warnings

import os
import time
import resource
import numpy as np
import pandas as pd
import yaml
# from memory_profiler import profile

import mne


# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Pipeline
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP

# Dimension reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Covariance estimation
from pyriemann.estimation import Covariances as pyr_cov

# Transformations
from pyriemann.tangentspace import TangentSpace
from geomstats.learning.preprocessing import ToTangentSpace as geo_ts

# Metrics
from geomstats.geometry.spd_matrices import SPDMetricLogEuclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric

# Classification algorithms
from pyriemann.classification import MDM
from sklearn.svm import SVC



# Framework
import moabb
from moabb.datasets import AlexMI, BNCI2014001, BNCI2014002, BNCI2014004, BNCI2015001, BNCI2015004, Cho2017, Lee2019_MI, MunichMI, Ofner2017, PhysionetMI, Schirrmeister2017, Shin2017A, Shin2017B, Weibo2014, Zhou2016
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation
from moabb.paradigms import MotorImagery
from utils import load_metrics
from itertools import product
from classifiers import classifiers

# MDM_classifiers = [MDM(metric=m) for m in metrics]

os.makedirs('results/', exist_ok=True)

paradigm = MotorImagery()
pipelines = {}
# datasets = [AlexMI()]
datasets = [AlexMI(), BNCI2014001(), BNCI2014002(), BNCI2014004(), BNCI2015001(), BNCI2015004(), Cho2017(), Lee2019_MI(), MunichMI(), Ofner2017(), PhysionetMI(), Schirrmeister2017(), Shin2017A(), Shin2017B(), Weibo2014(), Zhou2016()]

# projections = [None, TangentSpace, CustomToTangentSpace]
metrics = load_metrics()

for metric, classifier in product(metrics, classifiers):
    metric_name = metric.metric if isinstance(metric, TangentSpace) else metric.geometry_name
    name = f'{metric.__class__.__name__}_{metric_name}_{classifier}'
    print(name)
    pipelines[name] = make_pipeline(pyr_cov(), metric, classifier)

metrics = [
    'euclid',
    'riemann',
    'logeuclid',
    'logdet',
    'kullback_sym',
    'wasserstein',
]

for m in metrics:
    name = f'MDM_{m}'
    pipelines[name] = make_pipeline(pyr_cov(), MDM(metric=m))

evaluation_within_session = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets[:5],
    overwrite=True,
    hdf5_path=None,
)
results_within_session = evaluation_within_session.process(pipelines)


def plot(results, kind="bar", y="score", x="subject", hue="pipeline", palette="tab10", height=40):
    sns.set(font_scale=8)
    g = sns.catplot(
        kind=kind,
        y=y,
        x=x,
        hue=hue,
        data=results,
        #orient="h",
        palette=palette,
        height=height,
    )
    #g.set_xticklabels(plt.get_xticklabels(), fontsize = 18)
    plt.show()

results_within_session.to_csv('results/within_session.csv')
plot(results_within_session)
