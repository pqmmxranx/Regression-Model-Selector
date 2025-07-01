import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from typing import Union, Tuple

"""
Machine Learning Regression Model Selector

Regression Models:
    Multiple Linear Regression
    Polynomial Regression
    Support Vector Regression RBF
    Support Vector Regression Linear
    Support Vector Regression Polynomial
    Decision Tree Regression
    Random Forest Regression
    
Evaluates the dataset into each regression model and determines
the best model for the specific dataset based on its r squared
value.

28/06/2025
DD/MM/YYYY

Go to Line 355 to 357
"""

""" Customizable Constants """
TEST_SIZE = 0.2
RANDOM_STATE = 0
POLY_DEGREE = 4
N_ESTIMATORS = 10


def divider() -> None:
    print("*" * 80)


def tt_split(
        X: np.ndarray,
        y: np.ndarray
    ) -> list:
    """
    Splits the dataset into data fit for machine learning which are
    `X_train`, `X_test`, `y_train`, `y_test`.

    :param X: Independent variables
    :param y: Dependent variables
    :return: The training and test sets for the independent and
        dependent variables. Training: `X_train`, `y_train`.
        Test: `X_test`, `y_test`.
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


def print_details(
        ypred: np.ndarray,
        ytest: np.ndarray
    ) -> None:
    """
    Prints the predicted values (`ypred`) and the
    actual values ('ytest')

    :param ypred: Predicted values of the test set
    :param ytest: Actual values of the test set
    """
    np.set_printoptions(precision=2)
    y_reshaped = [i.reshape(len(i), 1) for i in (ypred, ytest)]

    print(np.concatenate((y_reshaped[0], y_reshaped[1]), 1))
    print(f"\t{r2_score(ytest, ypred)}")


def multiple_linear_regression(
        X: np.ndarray,
        y: np.ndarray,
        predict_values=None
    ) -> Union[float, Tuple[float, float], Tuple[float, str]]:
    """
    Applies the Multiple Linear Regression method from scikit-learn
    and evaluates its performance on the specific dataset.

    :param X: Independent variables
    :param y: Dependent variables
    :param predict_values: X values to be predicted by the model.
        Must be valid or else function will error.
    :return: If `predict_values` is None, function will only return
        Coefficient of determination (R squared value) of the
        specific regression model. If `predict_values` is not None and
        is valid, it will return R squared value and predicted value.
        If `predict_values` is not None and is invalid, it will return
        R squared value and string "Invalid prediction values".
    """
    Xtrain, Xtest, ytrain, ytest = tt_split(X, y)

    multiple_linear_regressor = LinearRegression()
    multiple_linear_regressor.fit(Xtrain, ytrain)

    ypred = multiple_linear_regressor.predict(Xtest)
    print("multiple_linear_regression")
    print_details(ypred, ytest)

    r2 = r2_score(ytest, ypred)

    if can_be_plotted and plot_list["Multiple Linear"]:
        plt.plot(X, multiple_linear_regressor.predict(X), color="blue")

    if predict_values:
        try:
            return (r2,
                    multiple_linear_regressor.predict(
                        [predict_values])[0])
        except Exception as e:
            print(f"Invalid prediction values. {e}")
            return r2, "Invalid prediction values"

    return r2


def polynomial_regression(
        X: np.ndarray,
        y: np.ndarray,
        predict_values=None
    ) -> Union[float, Tuple[float, float], Tuple[float, str]]:
    """
    Applies the Polynomial Regression method from scikit-learn and
    evaluates its performance on the specific dataset.

    :param X: Independent variables
    :param y: Dependent variables
    :param predict_values: X values to be predicted by the model.
        Must be valid or else function will error.
    :return: If `predict_values` is None, function will only return
        Coefficient of determination (R squared value) of the
        specific regression model. If `predict_values` is not None and
        is valid, it will return R squared value and predicted value.
        If `predict_values` is not None and is invalid, it will return
        R squared value and string "Invalid prediction values".
    """
    Xtrain, Xtest, ytrain, ytest = tt_split(X, y)

    poly_reg = PolynomialFeatures(
        degree=POLY_DEGREE
    )
    Xpoly = poly_reg.fit_transform(Xtrain)
    polynomial_regressor = LinearRegression()
    polynomial_regressor.fit(Xpoly, ytrain)

    ypred = polynomial_regressor.predict(poly_reg.transform(Xtest))
    print("polynomial_regression")
    print_details(ypred, ytest)

    r2 = r2_score(ytest, ypred)

    if can_be_plotted and plot_list["Polynomial"]:
        Xplot = poly_reg.fit_transform(X)
        plt.plot(X, polynomial_regressor.predict(Xplot), color="purple")

    if predict_values:
        try:
            transformed_predict = poly_reg.fit_transform(
                [predict_values])
            return (r2,
                    polynomial_regressor.predict(
                        transformed_predict)[0])
        except Exception as e:
            print(f"Invalid prediction values. {e}")
            return r2, "Invalid prediction values"

    return r2


def support_vector_regression(
        X: np.ndarray,
        y: np.ndarray,
        predict_values=None,
        kernel: str = 'rbf'
    ) -> Union[float, Tuple[float, float], Tuple[float, str]]:
    """
    Applies the Support Vector Regression method from scikit-learn and
    evaluates its performance on the specific dataset.

    :param X: Independent variables
    :param y: Dependent variables
    :param predict_values: X values to be predicted by the model.
        Must be valid or else function will error.
    :param kernel: SVR Kernel to be used from `rbf` (default),
        `linear`, `poly`, and `sigmoid` (rarely used).
    :return: If `predict_values` is None, function will only return
        Coefficient of determination (R squared value) of the
        specific regression model. If `predict_values` is not None and
        is valid, it will return R squared value and predicted value.
        If `predict_values` is not None and is invalid, it will return
        R squared value and string "Invalid prediction values".
    """
    y = y.reshape(len(y), 1)
    Xtrain, Xtest, ytrain, ytest = tt_split(X, y)

    scX = StandardScaler()
    scy = StandardScaler()
    Xtrain = scX.fit_transform(Xtrain)
    ytrain = scy.fit_transform(ytrain)

    support_vector_regressor = SVR(
        kernel=kernel
    )
    support_vector_regressor.fit(Xtrain, ytrain.ravel())

    pred_scale = support_vector_regressor.predict(scX.transform(Xtest))
    ypred = scy.inverse_transform(pred_scale.reshape(-1, 1))
    print(f"support_vector_regression {kernel}")
    print_details(ypred, ytest)

    r2 = r2_score(ytest, ypred)

    for kernel_t in ("RBF", "Linear", "Poly"):
        if can_be_plotted and plot_list[f"Support Vector {kernel_t}"]:
            pass

    if predict_values:
        try:
            transformed_predict = scX.transform([predict_values])
            predict_scale = support_vector_regressor.predict(
                transformed_predict)
            predict_scale_reshaped = predict_scale.reshape(-1, 1)
            return (r2,
                    scy.inverse_transform(
                        predict_scale_reshaped)[0][0])
        except Exception as e:
            print(f"Invalid prediction values. {e}")
            return r2, "Invalid prediction values"

    return r2


def decision_tree_regression(
        X: np.ndarray,
        y: np.ndarray,
        predict_values=None
    ) -> Union[float, Tuple[float, float], Tuple[float, str]]:
    """
    Applies the Decision Tree Regression method from scikit-learn and
    evaluates its performance on the specific dataset.

    :param X: Independent variables
    :param y: Dependent variables
    :param predict_values: X values to be predicted by the model.
        Must be valid or else function will error.
    :return: If `predict_values` is None, function will only return
        Coefficient of determination (R squared value) of the
        specific regression model. If `predict_values` is not None and
        is valid, it will return R squared value and predicted value.
        If `predict_values` is not None and is invalid, it will return
        R squared value and string "Invalid prediction values".
    """
    Xtrain, Xtest, ytrain, ytest = tt_split(X, y)

    decision_tree_regressor = DecisionTreeRegressor(
        random_state=RANDOM_STATE
    )
    decision_tree_regressor.fit(Xtrain, ytrain)

    ypred = decision_tree_regressor.predict(Xtest)
    print("decision_tree_regression")
    print_details(ypred, ytest)

    r2 = r2_score(ytest, ypred)

    if can_be_plotted and plot_list["Decision Tree"]:
        Xgrid = np.arange(min(X)[0], max(X)[0], 0.1)
        Xgrid = Xgrid.reshape((len(Xgrid), 1))
        X_grid_pred = decision_tree_regressor.predict(Xgrid)
        plt.plot(Xgrid, X_grid_pred, color='orange')

    if predict_values:
        try:
            return (r2,
                    decision_tree_regressor.predict(
                        [predict_values])[0])
        except Exception as e:
            print(f"Invalid prediction values. {e}")
            return r2, "Invalid prediction values"

    return r2


def random_forest_regression(
        X: np.ndarray,
        y: np.ndarray,
        predict_values=None
    ) -> Union[float, Tuple[float, float], Tuple[float, str]]:
    """
    Applies the Random Forest Regression method from scikit-learn and
    evaluates its performance on the specific dataset.

    :param X: Independent variables
    :param y: Dependent variables
    :param predict_values: X values to be predicted by the model.
        Must be valid or else function will error.
    :return: If `predict_values` is None, function will only return
        Coefficient of determination (R squared value) of the
        specific regression model. If `predict_values` is not None and
        is valid, it will return R squared value and predicted value.
        If `predict_values` is not None and is invalid, it will return
        R squared value and string "Invalid prediction values".
    """
    Xtrain, Xtest, ytrain, ytest = tt_split(X, y)

    random_forest_regressor = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE
    )
    random_forest_regressor.fit(Xtrain, ytrain)

    ypred = random_forest_regressor.predict(Xtest)
    print("random_forest_regression")
    print_details(ypred, ytest)

    r2 = r2_score(ytest, ypred)

    if can_be_plotted and plot_list["Random Forest"]:
        Xgrid = np.arange(min(X)[0], max(X)[0], 0.1)
        Xgrid = Xgrid.reshape((len(Xgrid), 1))
        Xgrid_pred = random_forest_regressor.predict(Xgrid)
        plt.plot(Xgrid, Xgrid_pred, color='green')

    if predict_values:
        try:
            return (r2,
                    random_forest_regressor.predict(
                        [predict_values])[0])
        except Exception as e:
            print(f"Invalid prediction values. {e}")
            return r2, "Invalid prediction values"

    return r2


def main():
    global can_be_plotted
    global plot_list

    csv_file = "stock_data"
    prediction_values = [12000]     # Example
    plot_list = {
        "Multiple Linear": False,
        "Polynomial": False,
        "Decision Tree": False,
        "Random Forest": False,
        "Support Vector RBF": False,        # DOESNT WORK
        "Support Vector Linear": False,     # DOESNT WORK
        "Support Vector Poly": True,        # DOESNT WORK
    }
    """ 
    Change `csv_file` to a specific file name for automatic execution.
    Default value is None or "".
    
    Change `prediction_values` into an array for predictions. Must be
    valid arrays catered for the dataset or else it will cause an error.
    For testing: 14.96,41.76,1024.07,73.17
    
    Change `plot_list` values if you want to graph the specific
    regression model. `True` if yes, `False` if no.
    """

    can_be_plotted = False

    if not csv_file:
        while True:
            """ Get valid csv file from input """
            global dataset
            try:
                csv_file = input("Input the name of the csv file: ")
                dataset = pd.read_csv(f"{csv_file}.csv")
                break
            except Exception as e:
                print(f"Invalid input. {e}")
    else:
        dataset = pd.read_csv(f"{csv_file}.csv")

    """
    Lines below only apply if the last column is the dependent variable
    
    default table:
    ===============================
    | IND 1 | IND 2 | IND X | DEP |
    ===============================
    
    Number of independent variables is variable.
    
    Edit dataset.iloc[*:*, *:*] if there are only specific columns that
    need to be changed 
    
    default values:
    X - [:, :-1]
    y - [:, -1]
    """
    X = dataset.iloc[
        :, :-1          # INDEPENDENT VARIABLE/S
    ].values

    y = dataset.iloc[
        :, -1           # DEPENDENT VARIABLE
    ].values

    if X.shape[0] == y.shape[0]:
        can_be_plotted = True

    """ Calling functions for each regression model """
    results_list = {
        "Multiple Linear":
            multiple_linear_regression(X, y, prediction_values),
        "Polynomial":
            polynomial_regression(X, y, prediction_values),
        "Decision Tree":
            decision_tree_regression(X, y, prediction_values),
        "Random Forest":
            random_forest_regression(X, y, prediction_values),
        "Support Vector RBF":
            support_vector_regression(X, y, prediction_values,
                                      kernel='rbf'),
        "Support Vector Linear":
            support_vector_regression(X, y, prediction_values,
                                      kernel='linear'),
        "Support Vector Poly":
            support_vector_regression(X, y, prediction_values,
                                      kernel='poly'),
    }

    divider()

    values_list = [value for value in results_list.values()]

    if not prediction_values:
        for key, value in results_list.items():
            print(f"{f'{key} Regression'}:")
            print(f"\tR squared value: {value}")

        """ 
        Get best result or highest r squared value 
        from `results_list` 
        """
        highest = [(key, value) for key, value in results_list.items() \
                   if value == max(values_list)][0]

        divider()

        print(f"Use {highest[0]} Regression with an R squared value "
              f"of {highest[1]}")
    else:
        for key, value in results_list.items():
            print(f"{f'{key} Regression'}:")
            print(f"\tR squared value: {value[0]}")
            print(f"\tPrediction: {value[1]}")

        """ 
        Get best result or highest r squared value and prediction 
        from `results_list` 
        """
        highest = [(key, value) for key, value in results_list.items() \
                   if value[0] == max(values_list)[0]][0]

        divider()

        print(f"Use {highest[0]} Regression with an R squared value "
              f"of {highest[1][0]} and prediction of {highest[1][1]}")

    if any(plot_list.values()):
        try:
            """ Plot samples in graph with matplotlib.pyplot """
            plt.scatter(X, y, color="red")
            plt.show()
        except ValueError:
            """ if X.shape != y.shape """
            print("Cannot be graphed. X and y should have the same "
                  "number of columns.")


if __name__ == '__main__':
    main()
