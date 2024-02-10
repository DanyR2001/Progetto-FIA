import random
from sklearn import linear_model #LinearRegression, SGDRegressor,Ridge, LARS, LassoLars, BayesianRidge, ARDRegression, LogisticRegression, TweedieRegressor
from sklearn import neighbors #KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn import gaussian_process #GaussianProcessRegressor
from sklearn import cross_decomposition #PLSRegression
from sklearn import svm #SVR, NuSVR, LinearSVR [R] regressor
from sklearn import tree #DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score, explained_variance_score
)
import statistics


class Agent:
    model = None
    name = None
    type = None
    n_job = None

    def __init__(self, model_type="Complement", n_job:int = 1,randState:int=random.randint(0, 256)) -> None:
        self.n_job=n_job
        self.randState=randState
        print("------------------------------ Start " + model_type + " ------------------------------")
        print("")
        if (model_type == "LinearRegression"):
            self.model=linear_model.LinearRegression(n_jobs=self.n_job)
            self.name=model_type
            self.type= "lineare"
        elif (model_type == "Ridge"):
            self.model = linear_model.Ridge()
            self.name = model_type
            self.type = "lineare"
        elif (model_type == "SGDRegressor"):
            self.model = linear_model.SGDRegressor()
            self.name = model_type
            self.type = "lineare"
        elif (model_type == "LARS"):
            self.model = linear_model.Lars()
            self.name = model_type
            self.type = "lineare"
        elif (model_type == "LassoLars"):
            self.model = linear_model.LassoLars()
            self.name = model_type
            self.type = "lineare"
        elif (model_type == "BayesianRidge"):
            self.model = linear_model.BayesianRidge()
            self.name = model_type
            self.type = "lineare"
        elif (model_type == "ARDRegression"):
            self.model = linear_model.ARDRegression()
            self.name = model_type
            self.type = "lineare"
        elif (model_type == "LogisticRegression"):
            self.model = linear_model.LogisticRegression(solver='liblinear', max_iter=3000,n_jobs=self.n_job)
            self.name = model_type
            self.type = "lineare"
        elif (model_type == "TweedieRegressor"):
            self.model = linear_model.TweedieRegressor(max_iter=5000)
            self.name = model_type
            self.type = "lineare"
        elif (model_type == "DecisionTreeRegressor"):
            self.model = tree.DecisionTreeRegressor()
            self.name = model_type
            self.type = "TreeRegressor"
        elif (model_type == "KNeighborsRegressor"):
            self.model = neighbors.KNeighborsRegressor(n_jobs=self.n_job)
            self.name = model_type
            self.type = "NeighborsRegressor"
        elif (model_type == "RadiusNeighborsRegressor"):
            self.model = neighbors.RadiusNeighborsRegressor(n_jobs=self.n_job)
            self.name = model_type
            self.type = "NeighborsRegressor"
        elif (model_type == "GaussianProcessRegressor"):
            self.model = gaussian_process.GaussianProcessRegressor()
            self.name = model_type
            self.type = "GaussianProcessRegressor"
        elif (model_type == "SVR"):
            self.model = svm.SVR()
            self.name = model_type
            self.type = "SVR"
        elif (model_type == "NuSVR"):
            self.model = svm.NuSVR()
            self.name = model_type
            self.type = "SVR"
        elif (model_type == "LinearSVR"):
            self.model = svm.LinearSVR(dual=True,max_iter=8000)
            self.name = model_type
            self.type = "SVR"
        elif (model_type == "RandomForestRegressor"):
            self.model=RandomForestRegressor(n_jobs=self.n_job)
            self.name = model_type
            self.type = "TreeRegressor"


    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def valuation(self, y_test, pred):
        variance_score = explained_variance_score(y_test, pred)
        mean_absolute = mean_absolute_error(y_test, pred)
        mean_squared = mean_squared_error(y_test, pred)
        root_mean_absolute = mean_squared_error(y_test, pred, squared=False)
        r2 = r2_score(y_test, pred)
        return variance_score, mean_absolute, mean_squared, root_mean_absolute, r2

    def cross_validation(self, X_train, y_train):
        rfk = RepeatedKFold(n_splits=10, n_repeats=4, random_state=self.randState)
        tests = list(["neg_mean_absolute_error", "neg_mean_squared_error","neg_root_mean_squared_error", "r2"])
        cv_score = cross_validate(self.model, X_train, y_train, cv=rfk, n_jobs=self.n_job, verbose=5, scoring=tests)
        fit_time_mean = statistics.mean(cv_score['fit_time'])
        score_time_mean = statistics.mean(cv_score['score_time'])
        absolute_error_mean = statistics.mean(cv_score['test_neg_mean_absolute_error'])
        squared_error_mean = statistics.mean(cv_score['test_neg_mean_squared_error'])
        root_mean_squared_error_mean = statistics.mean(cv_score['test_neg_root_mean_squared_error'])
        r2_mean = statistics.mean(cv_score['test_r2'])
        return fit_time_mean, score_time_mean, absolute_error_mean, squared_error_mean, root_mean_squared_error_mean,r2_mean