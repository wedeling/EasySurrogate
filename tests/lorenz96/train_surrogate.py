import easysurrogate as es
import numpy as np

lags = [range(1, 75)]
feats = ['X_data']

campaign = es.QSN_Campaign(load_data = True)