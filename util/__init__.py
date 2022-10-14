from .parser import Feature, MatrixFeature, StatsFeature, AnonymousSeqFeature


feature_extractor: Feature = AnonymousSeqFeature()
# feature_extractor = MatrixFeature()
# feature_extractor = StatsFeature()
