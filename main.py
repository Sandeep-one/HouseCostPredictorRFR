# pandas version 1.2.2
import pandas as pd
# numpyversion    1.19.2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn    version : 0.23.2
import sklearn.preprocessing as pre
import sklearn.model_selection as ms

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class MLException(Exception):  # class for Custom Exception
    pass


class CostPrediction:
    def __init__(self, data_set):
        # pd.set_option("display.max_columns", 20)
        df = pd.read_csv(data_set)

        self.__housing = df.drop("median_house_value", axis=1)
        self.__house_value = df["median_house_value"]
        self.__model = None
        self.__X_train = None
        self.__y_train = None
        self.__X_test = None
        self.__y_test = None
        self.__X_train_pipe = None
        self.__X_test_pipe = None
        self.__pipe_numeric = None
        self.__pipe_category = None
        self.__full_pipe = None

    def get_housing(self):
        return self.__housing

    def get_house_value(self):
        return self.__house_value

    def get_x_train(self):
        return self.__X_train

    def get_y_train(self):
        return self.__y_train

    def get_x_test(self):
        return self.__X_test

    def get_y_test(self):
        return self.__y_test

    def get_x_train_pipe(self):
        return self.__X_train_pipe

    def get_x_test_pipe(self):
        return self.__X_test_pipe

    def __set_train_test_split(self, a, b, c, d):
        self.__X_train = a.copy()
        self.__X_test = b.copy()
        self.__y_train = c.copy()
        self.__y_test = d.copy()

    def preprocessing(self):

        print("SPLITTING DATA FOR TRAINING AND TEST SET... \n")
        try:
            # self.__housing["rooms_per_household"] = self.__housing["total_rooms"] / self.__housing["households"]
            a, b, c, d = ms.train_test_split(self.get_housing(), self.get_house_value(), test_size=0.2, random_state=42)
            if a is None or b is None or c is None or d is None:
                raise MLException("\n !!!!! some error occur during train_test_split")
            self.__set_train_test_split(a, b, c, d)

            print("FEATURE SELECTION...\n")

            c1 = self.get_x_train().corr()

            self.__X_train["rooms_per_household"] = self.__X_train["total_rooms"] / self.__X_train["households"]
            self.__X_train.drop(["total_rooms", "total_bedrooms", "population"], axis=1, inplace=True)
            c2 = self.get_x_train().corr()

            print("Close the Graph for further proceeding")

            fig = plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            sns.heatmap(c1, annot=True).set_title("Before Feature Selectiion")
            plt.subplot(1, 2, 2)
            sns.heatmap(c2, annot=True).set_title("After Feature Selectiion")
            plt.tight_layout()
            plt.show()

            self.__X_test["rooms_per_household"] = self.__X_test["total_rooms"] / self.__X_test["households"]
            self.__X_test.drop(["total_rooms", "total_bedrooms", "population"], axis=1, inplace=True)

            self.__pipeline()
        except MLException as e:
            print(e)
            return False
        except Exception as e:
            print("\n !!!!! some error occur " + e)
            return False
        else:
            return True

    def __pipeline(self):
        lst_columns = ["longitude", "latitude", "housing_median_age",
                       "households", "median_income", "rooms_per_household"]

        print("\nCREATING PIPELINE FOR PRE-PROCESSING... \n")

        self.__pipe_numeric = Pipeline([
            ("SI", SimpleImputer(strategy="median")),
            ("SC", pre.StandardScaler())
        ])

        self.__pipe_category = Pipeline([
            ("SI", SimpleImputer(strategy="most_frequent")),
            ("OHE", pre.OneHotEncoder(sparse=False))
        ])

        self.__full_pipe = ColumnTransformer([
            ("Numeric", self.__pipe_numeric, lst_columns),
            ("category", self.__pipe_category, ["ocean_proximity"])
        ])

        # pipe = Pipeline([
        #     ("FULL_PIPE", full_pipe),
        #     ("PCA", PCA(n_components=2))
        # ])
        if self.__full_pipe is not None:

            self.__X_train_pipe = self.__full_pipe.fit_transform(self.get_x_train())
            self.__X_test_pipe = self.__full_pipe.transform(self.get_x_test())
        else:
            raise MLException("\n !!! Some Error occurred during pipeline process ")

    def random_forest(self):
        print("\nTRAINING MODEL FOR RANDOM FOREST... \n")
        model = RandomForestRegressor(random_state=1)
        # model.fit(self.get_x_train_pipe(), self.get_y_train())

        print("Hyper Parameter Tuning (please, wait for 2-3 minutes)... ")

        p = {"n_estimators": [100, 300],  # 100,
             "max_features": [4, 6],  # 4,
             'max_depth': [70, 80]}  # , 80

        self.__model = ms.GridSearchCV(model, param_grid=p, cv=2)
        self.__model.fit(self.get_x_train_pipe(), self.get_y_train())
        print("Best Parameters Selected : \n", self.__model.best_params_)
        print("\nBEST SCORES for Training SEt: ", self.__model.best_score_)

        s = self.__model.score(self.get_x_test_pipe(), self.get_y_test())
        print("\nBEST R sq. SCORES for Test SEt: ", s)
        r = mean_squared_error(self.get_y_test(), self.__model.predict(self.get_x_test_pipe()))
        print("\nRoot Mean Sq. Error for Test SEt: ", np.sqrt(r))

    def predict(self):

        print("Enter longitude:  ")
        lo = float(input())
        print("Enter latitude: ")
        la = float(input())
        print("Enter housing_median_age: ")
        hma = float(input())
        print("Enter households: ")
        hh = float(input())
        print("Enter median_income: ")
        mi = float(input())
        mi = mi / 10000
        print("Enter rooms_per_household: ")
        rph = float(input())

        print("choose for ocean_proximity\n ")

        while True:
            print(" 1 for, <1H OCEAN,"
                  " 2 for, INLAND,"
                  " 3 for, NEAR OCEAN,"
                  " 4 for, NEAR BAY,"
                  " 5 for, ISLAND,")
            op = int(input())
            if op == 1:
                op = "<1H OCEAN"
                break
            elif op == 2:
                op = "INLAND"
                break
            elif op == 3:
                op = "NEAR OCEAN"
                break
            elif op == 4:
                op = "NEAR BAY"
                break
            elif op == 5:
                op = "ISLAND"
                break
            else:
                print("Kindly, Enter Right Option")

        df = pd.DataFrame([[lo, la, hma, hh, mi, op, rph]],
                          columns=['longitude', 'latitude', 'housing_median_age', 'households',
                                   'median_income', 'ocean_proximity', 'rooms_per_household'])

        y = self.__model.predict(self.__full_pipe.transform(df))
        print("Predicted Price : ", y[0])


dataset = "housing.csv"
obj = CostPrediction(dataset)
print("\n\t\t RANDOM FOREST REGRESSOR FOT PREDICTING HOUSE PRICE\n\n")
preprocessing = obj.preprocessing()

if preprocessing:
    print("Press (y) For Random Forest Regression Model Training")
    n = input()
    flag = False

    if n == "Y" or n == "y":
        obj.random_forest()
        flag = True

        print("\nwant to predict cost (Y/N) ??")
        n = input()
        if n == "Y" or n == 'y':
            obj.predict()
        else:
            print("byee...")
    else:
        print("Invalid Choice...\n")
        print("Re Run the Program...")

else:
    print("Check Your Program")
