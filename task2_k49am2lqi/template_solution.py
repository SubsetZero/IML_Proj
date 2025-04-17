# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables   

    train_df = pd.get_dummies(train_df, columns=['season'], prefix='season') 
    test_df = pd.get_dummies(test_df, columns=['season'], prefix='season')

    train_df = train_df.dropna(subset=['price_CHF'])

    imp = IterativeImputer()
    train_df_drop = train_df.drop(columns=['price_CHF']) # Drop the target variable from the training data
    train_df_imp = pd.DataFrame(imp.fit_transform(train_df_drop), columns=train_df_drop.columns) #Analize and transform data with imputer
    test_df = pd.DataFrame(imp.transform(test_df), columns=test_df.columns) # Transform test data with imputer (no fit)

    X_train = train_df_imp.values
    y_train = train_df['price_CHF'].values
    X_test = test_df.values

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        #TODO: Define the model and fit it using (X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train

        kernels = [
            ("RBF", RBF()),
            ("Matern", Matern()),
            ("RationalQuadratic", RationalQuadratic()),
            ("DotProduct", DotProduct())
        ]

        best_score = -np.inf
        best_kernel = None
        best_model = None

        for name, kernel in kernels:
            gpr = GaussianProcessRegressor(kernel=kernel)
            
            scores = cross_val_score(gpr, X_train, y_train, cv=5, scoring='r2')
            mean_score = scores.mean()
            print(f"Kernel: {name}, Mean R²: {mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_kernel = name
                best_model = gpr

        print(f"Best kernel selected: {best_kernel} with R² = {best_score:.4f}")

        # Refit on all training data with best kernel
        best_model.fit(X_train, y_train)
        self._model = best_model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred=np.zeros(X_test.shape[0])
        #TODO: Use the model to make predictions y_pred using test data X_test
        y_pred = self._model.predict(X_test)

        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    model = Model()
    # Use this function for training the model
    model.train(X_train=X_train, y_train=y_train)
    # Use this function for inferece
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

