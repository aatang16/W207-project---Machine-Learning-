import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class FeatureMaker:

    '''
    Class to do data cleaning and feature engineering. Takes the paths to the
    training and test sets a arguments and produces frames with the necessary
    features for the classifiers which we plan to use.
    '''

    def __init__(self, training, test, test_size, scale=False):

        '''
        Initializes with training and test set .csv files. Pass in a Path obj
        '''

        self.training_set = pd.read_csv(training)
        self.test_set = pd.read_csv(test)
        self.test_size = test_size
        self.scale = scale

    def describe_data(self):

        '''
        Method to explore the data a little bit to verify structure
        '''
        pd.set_option('display.max_columns', None)

        print(self.training_set.describe())
        print(self.test_set.describe())

        return

    def groom_data(self):

        '''
        Clean the data and simplify some of the features.
        '''

        # Combine Horizontal and Vertical Distance to Hydrology (Distance)
        # c**2 = a**2 + b**2

        vertical = self.training_set[
                'Vertical_Distance_To_Hydrology'
                ].to_numpy()
        horiz = self.training_set[
                'Horizontal_Distance_To_Hydrology'
                ].to_numpy()
        distance = np.sqrt(np.square(horiz) + np.square(vertical))

        self.training_set['Distance_To_Hydrology'] = distance

        # Drop the Horiz/Vert Distance to Hydrology features

        self.training_set = self.training_set.drop([
            'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology'
            ], axis=1)

        # Drop empty soil type columns

        self.training_set = self.training_set.drop([
            'Soil_Type7',
            'Soil_Type15'
            ], axis=1)

        # Drop the Id column from the dataframes. Not needed

        self.X = self.training_set.drop(['Id'], axis=1)
        self.X = self.training_set.drop(['Cover_Type'], axis=1)
        self.y = self.training_set['Cover_Type']

        # Train test split

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=self.test_size)

        # Define the numerical columns

        numerical_cols = [
                'Elevation',
                'Aspect',
                'Slope',
                'Distance_To_Hydrology',
                'Horizontal_Distance_To_Roadways',
                'Hillshade_9am',
                'Hillshade_Noon',
                'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points',
                ]

        # Standard Scale the numerical columns if self.scale is true

        if self.scale:
            # scaler = StandardScaler()
            # self.X_train[numerical_cols] = scaler.fit_transform(
            #        self.X_train[numerical_cols])

            scaler = MinMaxScaler()
            self.X_train[numerical_cols] = scaler.fit_transform(
                    self.X_train[numerical_cols]
                    )

        # Return values

        return self.X_train, self.X_test, self.y_train, self.y_test
