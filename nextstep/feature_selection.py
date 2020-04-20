import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE


class feature_selection:
    """feature selection calss
    """
    
    def __init__(self):
        print('successfully substantiated the feature_selection class')


    ###############
    # SelectKBest #
    ###############


    def get_selector_from_SelectKBest(self, score_func, X, y, num_features=10):
        """a filter based feature selection method where the user specifies a metric and uses that to filter features

        :param score_func: a function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores. For a list of available score_func, refer to the "See also" section on https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html.
        :param X: the training input samples
        :type X: pandas DataFrame of shape (n_samples, n_features)
        :param y: the target values
        :type y: array-like of shape (n_samples,)
        :param num_features: number of top features to select
        :type num_features: int, optional (default=10)
        :return: the fitted SelectKBest selector
        """

        # fit the a model with score_func
        selector = SelectKBest(score_func, k=num_features)
        selector.fit(X, y)

        return selector



    def get_features_from_SelectKBest(self, score_func, X, y, num_features=10, show_plot=True):
        """a filter based feature selection method where the user specifies a metric and uses that to filter features; also, this function displays the SelectKBest result with a plot, which facilitates identification of the "elbow" point   

        :param score_func: a function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores. For a list of available score_func, refer to the "See also" section on https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html.
        :param X: the training input samples
        :type X: pandas DataFrame of shape (n_samples, n_features)
        :param y: the target values
        :type y: array-like of shape (n_samples,)
        :param num_features: number of top features to select
        :type num_features: int, optional (default=10)
        :param show_plot: if set to True, a plot of the scores of the k best features is displayed. 
        :type show_plot: boolean, optional (default=True)
        :return: a pandas Dataframe containing the best k features, sorted by score in descending order
        """

        # get the selector for SelectKBest
        selector = self.get_selector_from_SelectKBest(score_func, X, y, num_features=num_features)    

        # store the results in dataframes
        scores = pd.DataFrame(selector.scores_)
        columns = pd.DataFrame(X.columns)

        # concat the two dataframes
        scores_df = pd.concat([columns, scores], axis=1)
        scores_df.columns = ['Feature','Score']

        # rank and sort the features based on their scores
        scores_df = scores_df.nlargest(num_features,'Score').reset_index().drop('index',axis=1)

        # visualize scores_df
        if show_plot == True:
            plt.plot(scores_df['Feature'], scores_df['Score'], marker='o')
            plt.xticks(rotation='vertical')
            plt.xlabel("feature", fontweight="bold")
            plt.ylabel("score", fontweight="bold")
            plt.title("Best {} features from SelectKBest".format(num_features), fontweight="bold")
            plt.show()

        return scores_df




    #################################
    # Recursive Feature Elimination #
    #################################


    def get_selector_from_RFE(self, estimator, X, y, num_features=None, step=1, verbose=0):
        """
        a wrapper based feature selection method that considers the selection of a set of features as a search problem. The algorithm selects features by recursively considering smaller and smaller sets of features.

        :param estimator: an sklearn supervised learning estimator with a fit method and a coef_ attribute or through a feature_importances_ attribute.
        :param X: the training input samples
        :type X: pandas DataFrame of shape (n_samples, n_features)
        :param y: the target values
        :type y: array-like of shape (n_samples,)
        :param num_features: number of features to select. If None, half will be selected.
        :type num_features: int or None, optional (default=None)
        :param step: If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.
        :type step: int or float, optional (default=1)
        :param verbose: controls verbosity of output
        :type verbose: int, (default=0)
        :return: the fitted RFE selector
        """

        # fit the a model with estimator
        rfe_selector = RFE(estimator, n_features_to_select=num_features, step=step, verbose=verbose)
        rfe_selector.fit(X, y)

        return rfe_selector



    def get_features_from_RFE(self, estimator, X, y, num_features=None, step=1, verbose=0):
        """
        a wrapper based feature selection method that considers the selection of a set of features as a search problem. The algorithm selects features by recursively considering smaller and smaller sets of features.

        :param estimator: an sklearn supervised learning estimator with a fit method and a coef_ attribute or through a feature_importances_ attribute.
        :param X: the training input samples
        :type X: pandas DataFrame of shape (n_samples, n_features)
        :param y: the target values
        :type y: array-like of shape (n_samples,)
        :param num_features: number of features to select. If None, half will be selected.
        :type num_features: int or None, optional (default=None)
        :param step: If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.
        :type step: int or float, optional (default=1)
        :param verbose: controls verbosity of output
        :type verbose: int, (default=0)
        :return: a list containing the best n features
        """

        # get the RFE selector
        rfe_selector = self.get_selector_from_RFE(estimator, X, y, num_features=num_features, step=step, verbose=verbose)

        # obtain a list of selected features
        rfe_support = rfe_selector.get_support()
        rfe_features = X.loc[:,rfe_support].columns.tolist()

        return rfe_features




    ###################################################
    # Majority voting combining all feature selectors #
    ###################################################


    def select_features_by_majority_voting(self, X, selectors_dict):  
        """
        combines the various feature selection tools and use majority voting to decide which features to keep. Only works on selectors with a get_support() function

        :param X: the training input samples
        :type X: pandas DataFrame of shape (n_samples, n_features)
        :param selectors_dict: a dictionary with key being the name of a feature selection method, value being a selector object
        :type selectors:dict: Python dictionary
        :return: a pandas DataFrame ranked by the number of times a feature has been seleced by the various selectors; boolean columns indicating whether a feature has been selected by a particular selector.
        """

        for method, selector in selectors_dict.items():
            selectors_dict[method] = selector.get_support()

        feature_selection_df = pd.DataFrame(selectors_dict)
        feature_selection_df['Feature'] = X.columns
        cols = feature_selection_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        feature_selection_df = feature_selection_df[cols]

        # count number of times each feature is selected
        feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)

        # sort and display the feature selection dataframe
        feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
        feature_selection_df.index = range(1, len(feature_selection_df)+1)

        return feature_selection_df


