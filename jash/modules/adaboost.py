import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier


class ADA:

    '''
    ADABoost Classifier implementation. Input is splitted data
    '''

    def __init__(self, X_train, X_test, y_train, y_test):

        '''
        Inititlaize with needed parameters. Make them into numpy arrays
        for faster execution and lower memory overhead
        '''

        self.X_train = X_train.to_numpy()
        self.X_test = X_test.to_numpy()
        self.y_train = y_train.to_numpy()
        self.y_test = y_test.to_numpy()

    def describe_data(self):

        '''
        Small method just to describe that data that we have. Shows shapes
        '''

        X_train_shape = self.X_train.shape
        X_test_shape = self.X_test.shape
        y_train_shape = self.y_train.shape
        y_test_shape = self.y_test.shape

        print('\n')
        print(f'X Train Shape: {X_train_shape}')
        print(f'X Test Shape: {X_test_shape}')
        print(f'Y Train Shape: {y_train_shape}')
        print(f'Y Test Shape: {y_test_shape}')

        return

    def execute_classifier(self):

        clf = AdaBoostClassifier(n_estimators=600)
        clf.fit(self.X_train, self.y_train)

        self.y_pred = clf.predict(self.X_test)

    def get_metrics(self):

        accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        mean_abs_err = metrics.mean_absolute_error(self.y_test, self.y_pred)
        mean_sq_err = metrics.mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_sq_err)
        f1_score = metrics.f1_score(self.y_test,
                                    self.y_pred, average='weighted')

        print('\n')
        print('---------------------------------')
        print(f'|               Accuracy : {accuracy:0.3f}|')
        print(f'|    Mean Absolute Error : {mean_abs_err:0.3f}|')
        print(f'|     Mean Squared Error : {mean_sq_err:0.3f}|')
        print(f'|Root Mean Squared Error : {rmse:0.3f}|')
        print(f'|               F1 Score : {f1_score:0.3f}|')
        print('---------------------------------')

        return
