from sklearn.base import BaseEstimator, TransformerMixin

from geomstats.learning.preprocessing import ToTangentSpace
from geomstats.geometry.spd_matrices import SPDMetricLogEuclidean, SPDMetricEuclidean, SPDMetricBuresWasserstein, SPDMetricAffine
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.invariant_metric import BiInvariantMetric


class CustomToTangentSpace(BaseEstimator, TransformerMixin):

    def __init__(self, geometry_name):
        geometries = {
            'euclid': SPDMetricEuclidean,
            'logeuclid': SPDMetricLogEuclidean,
            'wasserstein': SPDMetricBuresWasserstein,
            'affine': SPDMetricAffine,
            'riemannian': RiemannianMetric,
            'biinvariant': BiInvariantMetric,
        }
        self.geometry_name = geometry_name
        self.geometry = geometries[geometry_name]
        self.tangent_space = None

    def fit(self, X, y=None, weights=None, base_point=None):
        n = X.shape[1]
        self.tangent_space = ToTangentSpace(geometry=self.geometry(n))
        return self.tangent_space.fit(X, y=y, weights=weights, base_point=base_point)

    def transform(self, X, base_point=None):
        return self.tangent_space.transform(X, base_point=base_point)
