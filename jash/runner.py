from pathlib import Path
from modules.nb import NB
from modules.rfc import RFC
from modules.knn import KNN
from modules.lgbm import LGBM
from modules.xg_boost import XG
from modules.adaboost import ADA
from modules.extra_trees import ETC
from modules.data_splitter import FeatureMaker

TRAINING_PATH = Path('./data/train.csv')
TEST_PATH = Path('./data/test.csv')

feature_obj = FeatureMaker(TRAINING_PATH, TEST_PATH, 0.1, scale=False)
X_train, X_test, y_train, y_test = feature_obj.groom_data()

print('\n--------------BUILDING RFC----------------')

rfc_obj = RFC(X_train, X_test, y_train, y_test)
rfc_obj.execute_classifier()
rfc_obj.get_metrics()

print('\n--------------BUILDING XGB----------------')

xg_obj = XG(X_train, X_test, y_train, y_test)
xg_obj.execute_classifier()
xg_obj.get_metrics()

print('\n--------------BUILDING LGB----------------')

lgbm_obj = LGBM(X_train, X_test, y_train, y_test)
lgbm_obj.execute_classifier()
lgbm_obj.get_metrics()

print('\n--------------BUILDING ETC----------------')

etc_obj = ETC(X_train, X_test, y_train, y_test)
etc_obj.execute_classifier()
etc_obj.get_metrics()

print('\n--------------BUILDING KNN----------------')
knn_obj = KNN(X_train, X_test, y_train, y_test)
knn_obj.execute_classifier()
knn_obj.get_metrics()

print('\n--------------BUILDING ADA----------------')
ada_obj = ADA(X_train, X_test, y_train, y_test)
ada_obj.execute_classifier()
ada_obj.get_metrics()

print('\n--------------BUILDING NBC----------------')
nb_obj = NB(X_train, X_test, y_train, y_test)
nb_obj.execute_classifier()
nb_obj.get_metrics()
