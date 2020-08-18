from .base import BaseProblem, maybe_int, maybe_float
from typing import Union

import pandas as pd
import numpy as np

try:
    import pynisher
except:
    pass
import xgboost
import os

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, BaseCrossValidator
from sklearn.preprocessing import FunctionTransformer, Normalizer, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


class SteelFoldPlate(BaseProblem):

    def __init__(self, folder, naive=False):
        self.folder = folder
        self.data_X, self.data_y = data_to_X_and_y(load_data(folder))
        self.naive = naive
        # TODO: Allow this to be picked?
        self.validator = StratifiedKFold()
        # self.validator = LeaveOneOut()
        self.argspec = all_args_spec()
        self.lbs_v, self.ubs_v, self.vartype_v, self.deps_v = argspec_to_vecs(self.argspec)
        self.random_state = 0

    def evaluate(self, x):
        time_limit_in_s = 8
        argdict = argspec_and_vec_to_argdict(self.argspec, x)
        # print(argdict)
        classifier = construct_classifier(argdict, self.random_state)
                
        try:
            evaluate_classifier_b = pynisher.enforce_limits(wall_time_in_s=time_limit_in_s)(evaluate_classifier)
        except:
            print("WARNING: Could not enforce limits on evaluate_classifier. Dropping limits.")
            evaluate_classifier_b = evaluate_classifier
        
        try:
            # evaluation is higher is better. But optimizer minimizes.
            # Flip sign to compensate.
            return -1 * evaluate_classifier_b(classifier, self.validator, self.data_X, self.data_y)
        except:
            return 0.0

    def lbs(self):
        return self.lbs_v

    def ubs(self):
        return self.ubs_v

    def vartype(self):
        return self.vartype_v

    def dims(self):
        return len(self.argspec)

    def dependencies(self):
        if self.naive:
            return super().dependencies()
        else:
            return self.deps_v

    def __str__(self):
        return f"SteelFoldPlate()"

def load_data(directory):
    # File containing the header names
    headerfile = os.path.join(directory, "Faults27x7_var")
    # File containing the actual data points
    datafile = os.path.join(directory, "Faults.NNA")

    # In order to work, the files should exist
    assert os.path.exists(headerfile)
    assert os.path.exists(datafile)

    # Parse headers
    with open(headerfile) as f:
        headers = [line.strip() for line in f.readlines()]
    
    # Construct dataframe with the right headers
    data = pd.read_table(datafile, header=0, names=headers)

    return data

def data_to_X_and_y(data):
    # Binary features indicating which one is the right class.
    class_headers = np.array(['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'])
    
    # Turn features into an array
    X = np.asarray(data.drop(class_headers, axis=1))
    # And classes into an array of strings.
    y = class_headers[np.argmax(np.asarray(data[class_headers]), axis=1)]

    return X, y

def evaluate_classifier(classifier: Pipeline, validator: BaseCrossValidator, X, y):

    scores = []

    for train_index, test_index in validator.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        # TODO: Something with dropout and the `max_tree` parameter. 
        y_pred_test = classifier.predict(X_test)
        # print(list(zip(y_pred_test, y_test)))
        scores.append(np.sum(y_test == y_pred_test) / y_test.shape[0])

    # Return mean score
    return np.mean(scores)

def argspec_to_vecs(argspec):
    name_to_idx = { k: i for (i, (k, v)) in enumerate(argspec.items()) }
    lbs = np.asarray([v['lb'] for (k, v) in argspec.items()])
    ubs = np.asarray([v['ub'] for (k, v) in argspec.items()])
    ty = np.asarray([v['type'] for (k, v) in argspec.items()])
    deps = np.asarray([ 
        {
            'on': name_to_idx[v['dependent']['on']], 
            'values': v['dependent']['values'] 
        } if v.get('dependent') is not None else None 
        for (k, v) in argspec.items() ])
    return lbs, ubs, ty, deps

def argspec_and_vec_to_argdict(argspec, vec):
    return dict(zip(argspec, vec))

# Parameters are listed beyond this point.
# Features without any change in value stay as-is.

def all_args_spec():
    args_preprocessing = preprocessing_args_spec()
    args_xgboost = xgboost_args_spec()
    
    args_all = dict()
    args_all.update(args_preprocessing)
    args_all.update(args_xgboost)

    return args_all

def construct_classifier(args, random_state):
    return make_pipeline(
        construct_preprocessing(args), 
        construct_xgboost(args, random_state)
    )

# - Preprocessing via scikit-learn
def preprocessing_args_spec():
    pp_argspec = {
        # Specifically for normalizer..
        # Corresponding to ['L1', 'L2', 'max']
        'pp_normalizer_norm': {'lb': 0, 'ub': 2, 'type': 'cat', 'default': 1},
        # 
        'pp_pca_whiten': {'lb': 0, 'ub': 1, 'type': 'cat', 'default': 0}, 
        # While a lower bound of zero is possible and is equivalent to dropping everything
        # it will cause the downstream classifier to fail.
        # TODO: Hyperopt-sklearn uses
        #       hp.qloguniform(np.log(0.51), np.log(30.5), 1.0)
        # Which ranges from 4-120
        # Note: default is set to 27 as this is min(n_samples, n_features) for the used dataset.
        'pp_pca_n_components': {'lb': 1, 'ub': 27, 'type': 'int', 'default': 27}
    } 
    # Per feature
    pp_argspec.update({
        # Corresponding to ['None', 'PCA', 'MinMaxScaler', 'Normalizer', 'StandardScaler']
        f'pp_kind_{fi}': {'lb': 0, 'ub': 4, 'type': 'cat', 'default': 4}
        for fi in range(0, 27)
    })
    # Scalers are per-feature configured as well
    pp_argspec.update({
        # Corresponding to ['None', 'PCA', 'MinMaxScaler', 'Normalizer', 'StandardScaler']
        f'pp_min_max_min_{fi}': {'lb': -1, 'ub': 0, 'type': 'cat', 'default': 0, 
            'dependent': {'on': f'pp_kind_{fi}', 'values': {2}}}
        for fi in range(0, 27)
    })
    pp_argspec.update({
        # Corresponding to ['None', 'PCA', 'MinMaxScaler', 'Normalizer', 'StandardScaler']
        f'pp_ss_mean_{fi}': {'lb': 0, 'ub': 1, 'type': 'cat', 'default': 1,
            'dependent': {'on': f'pp_kind_{fi}', 'values': {4}}}
        for fi in range(0, 27)
    })
    pp_argspec.update({
        # Corresponding to ['None', 'PCA', 'MinMaxScaler', 'Normalizer', 'StandardScaler']
        f'pp_ss_std_{fi}': {'lb': 0, 'ub': 1, 'type': 'cat', 'default': 1,
            'dependent': {'on': f'pp_kind_{fi}', 'values': {4}}}
        for fi in range(0, 27)
    })

    return pp_argspec

def construct_preprocessing(args):
    # Construct a more advanced preprocessing setup.
    features_pca = []
    features_norm = []
    transformers = []
    features_pass = []

    # Scalers & collect lists for other transformers.
    for fi in range(0, 27):
        kind = int(args[f'pp_kind_{fi}'])
        if kind == 0:
            features_pass.append(fi)
        elif kind == 1:
            features_pca.append(fi)
        elif kind == 2:
            min_range = float(args[f'pp_min_max_min_{fi}'])
            max_range = 1.0
            transformers.append(
                (f'scaler_{fi}', 
                MinMaxScaler(feature_range=(min_range, max_range)), 
                [fi]))
        elif kind == 3:
            features_norm.append(fi)
        elif kind == 4:
            with_mean = int(args[f'pp_ss_mean_{fi}']) == 1
            with_std = int(args[f'pp_ss_std_{fi}']) == 1
            transformers.append(
                (f'scaler_{fi}', 
                StandardScaler(with_mean=with_mean, with_std=with_std), 
                [fi]))

    # PCA
    if len(features_pca) > 0:
        pca_whiten = int(args['pp_pca_whiten']) == 1
        pca_n_components = np.minimum(int(args['pp_pca_n_components']), len(features_pca))
        transformers.append(('pca', PCA(n_components=pca_n_components, whiten=pca_whiten), features_pca))
    
    # Normalizer
    if len(features_norm) > 0:
        normalizer_norm = param_preprocessing_normalizer_norm(int(args['pp_normalizer_norm']))
        transformers.append(('norm', Normalizer(norm=normalizer_norm), features_norm))

    # Do nothing
    if len(features_pass) > 0:
        transformers.append(('nop', FunctionTransformer(), features_pass))

    return ColumnTransformer(transformers)

def param_preprocessing_normalizer_norm(norm: int):
    options = [
        'l1',
        'l2',
        'max'
    ]
    return options[norm]

# - XGBoost
#  https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters

def xgboost_args_spec():
    dep_is_tree = {'on': 'xg_booster', 'values': {1, 2}}
    dep_is_dart = {'on': 'xg_booster', 'values': {1}}
    dep_is_gblinear = {'on': 'xg_booster', 'values': {0}}

    return {
        # 'xg_objective': {'lb': 0, 'ub': 3, 'type': 'cat'},
        # Default is 2 (gbtree)
        'xg_booster': {'lb': 0, 'ub': 2, 'type': 'cat', 'default': 2 },
        'xg_tree_method': {'lb': 0, 'ub': 3, 'type': 'cat', 'default': 0,
            'dependent': dep_is_tree},
        # TODO: [hyperopt-sklearn] has set this variable to be loguniform
        #       hp.loguniform(name, np.log(0.0001), np.log(0.5)) - 0.0001
        #       Under https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster
        #       however, `eta` (or alias `learning_rate`) is stated to have a range of [0, 1], as used below.
        #       Which one should be used?
        # 'xg_learning_rate': {'lb': 0, 'ub': 1, 'type': 'cont', 'default': 0.3},
        'xg_learning_rate': {'lb': np.log(0.0001), 'ub': np.log(0.5), 'type': 'cont', 'default': np.log(0.3+0.0001),
            'dependent': dep_is_tree},
        # NOTE: Arbitrarily cut off at 10: does not have an upper bound.
        'xg_gamma': {'lb': 0, 'ub': 10, 'type': 'cont', 'default': 0.0, 
            'dependent': dep_is_tree},
        # NOTE: Arbitrarily cut off at 10: does not have an upper bound.
        'xg_min_child_weight': {'lb': 0, 'ub': 10, 'type': 'int', 'default': 1,
            'dependent': dep_is_tree},
        # NOTE: Arbitrarily cut off at 10: does not have an upper bound.
        'xg_max_delta_step': {'lb': 0, 'ub': 10, 'type': 'int', 'default': 0,
            'dependent': dep_is_tree},
        # NOTE: Lower bound is 0 for the 4 below, non inclusive.
        # Set slightly higher to avoid issues around this bound.
        # TODO: [hyperopt-sklearn] has set the lower bound of these variables to 0.5
        #       But during optimization I have seen some of these take up values of 0.37
        #       for the best solution
        'xg_subsample': {'lb': 0.001, 'ub': 1.0, 'type': 'cont', 'default': 1.0, 'dependent': dep_is_tree},
        'xg_colsample_bytree': {'lb': 0.001, 'ub': 1.0, 'type': 'cont', 'default': 1.0, 'dependent': dep_is_tree},
        'xg_colsample_bylevel': {'lb': 0.001, 'ub': 1.0, 'type': 'cont', 'default': 1.0, 'dependent': dep_is_tree},
        'xg_colsample_bynode': {'lb': 0.001, 'ub': 1.0, 'type': 'cont', 'default': 1.0, 'dependent': dep_is_tree},
        # Original: upper bound is set arbitrarily at 10 (there is no real upper bound)
        # 'xg_alpha': {'lb': 0.0, 'ub': 10.0, 'type': 'cont', 'default': 0.0},
        # Instead we use the bounds used by hyperopt-sklearn:
        # [hyperopt-sklearn] uses hp.loguniform(name, np.log(0.0001), np.log(1)) - 0.0001
        # As such we set the uniform bounds to `np.log(0.0001)` and `np.log(1)`
        # and perform np.exp(xg_alpha) - 0.0001 in the function below.
        # TODO: Native passthrough for hyperopt / appoaches that have native support
        #       for these distributions (as they potentially make use of this information)
        'xg_alpha': {'lb': np.log(0.0001), 'ub': np.log(1), 'type': 'cont', 'default': np.log(0.0 + 0.0001)},
        # Original: Same as alpha, but 0 is excluded as well.
        # 'xg_lambda': {'lb': 0.001, 'ub': 10.0, 'type': 'cont', 'default': 1.0},
        # [hyperopt-sklearn] uses hp.loguniform(name, np.log(1), np.log(4))
        'xg_lambda': {'lb': np.log(1), 'ub': np.log(4), 'type': 'cont', 'default': np.log(1)},
        # Reweighting factor strongly depends on data
        # Alternative would be '1' or #neg/#pos
        # Given multiclass nature a tad difficult.
        # 'scale_pos_weight': {'lb': 0.0, 'ub': 1.0, 'type': 'cont'}
        # NOTE: Very strong relation to computational time.
        # Higher = better, but also more computationally intensive.
        # Upper bound is set arbitrarily at 10.
        # Maybe add a computational time limit?
        # Otherwise maybe better suited for a multi-objective problem.
        'xg_num_round': {'lb': 1, 'ub': 200, 'type': 'int', 'default': 100},
        # Maximum depth of a tree.
        # NOTE: Arbitrarily capped at 11, but the complexity scales exponentially
        # with the depth. As such this is arguably reasonable compared to the default of 6.
        # Extra: value was originally capped at 10, new value via [hyperopt-sklearn]
        'xg_max_depth': {'lb': 1, 'ub': 11, 'type': 'int', 'default': 6, 'dependent': dep_is_tree}, 

        # The following arguments are not directly accessible via the
        # SKLearn API. So whether these work or not, is a bit of a guess.
        # NOTE: Bounds are (0, 1), both non-inclusive.
        # Set higher and lower respectively to avoid issues. 
        'xg_sketch_eps': {'lb': 0.001, 'ub': 0.999, 'type': 'cont', 'default': 0.03,
            # xg_tree_method(2) = approx 
            'dependent': {'on': 'xg_tree_method', 'values': {2}}},
        'xg_grow_policy': {'lb': 0, 'ub': 1, 'type': 'cat', 'default': 0,
            # xg_tree_method(3) = hist 
            'dependent': {'on': 'xg_tree_method', 'values': {3}}},
        # NOTE: Arbitrarily capped at 128. 0 is a special value (no maximum)
        # Documentation states only relevant when value is lossguide.
        'xg_max_leaves': {'lb': 0, 'ub': 128, 'type': 'int', 'default': 0,
            # xg_grow_policy(1) = lossguide 
            'dependent': {'on': 'xg_grow_policy', 'values': {1}}},
        'xg_normalize_type': {'lb': 0, 'ub': 1, 'type': 'cat', 'default': 0,
            'dependent': dep_is_dart},
        'xg_rate_drop': {'lb': 0, 'ub': 1, 'type': 'cont', 'default': 0.0,
            'dependent': dep_is_dart},
        'xg_one_drop': {'lb': 0, 'ub': 1, 'type': 'cat', 'default': 0,
            'dependent': dep_is_dart},
        'xg_skip_drop': {'lb': 0, 'ub': 1, 'type': 'cont', 'default': 0.0,
            'dependent': dep_is_dart},
        'xg_updater': {'lb': 0, 'ub': 1, 'type': 'cat', 'default': 0,
            'dependent': dep_is_gblinear},
        'xg_feature_selector': {'lb': 0, 'ub': 4, 'type': 'cat', 'default': 0,
            'dependent': dep_is_gblinear},
        # NOTE: Bound arbitrarily set at 10, there is no real
        # upper bound.
        # Special case: 0 is select all.
        # Suggestion: Maybe split up
        # (select all: y/n, use 2nd integer if n)
        'xg_top_k': {'lb': 0, 'ub': 10, 'type': 'int', 'default': 0,
            'dependent': dep_is_gblinear},
    }

def construct_xgboost(args: dict, random_state):
    n_jobs = 1

    n_estimators = int(args['xg_num_round'])
    # objective = param_xgboost_objective(args['xg_objective'])
    objective = 'multi:softmax'
    booster = param_xgboost_booster(int(args['xg_booster']))
    tree_method = param_xgboost_tree_tree_method(maybe_int(args['xg_tree_method']))
    learning_rate = param_xgboost_learning_rate(maybe_float(args['xg_learning_rate']))
    gamma = maybe_float(args['xg_gamma'])
    
    max_depth = maybe_int(args['xg_max_depth'])
    min_child_weight = maybe_int(args['xg_min_child_weight'])
    max_delta_step = maybe_int(args['xg_max_delta_step'])
    
    subsample = maybe_float(args['xg_subsample'])
    colsample_bytree = maybe_float(args['xg_colsample_bytree'])
    colsample_bylevel = maybe_float(args['xg_colsample_bylevel'])
    colsample_bynode = maybe_float(args['xg_colsample_bynode'])
    
    reg_alpha = param_xgboost_alpha(float(args['xg_alpha']))
    reg_lambda = param_xgboost_lambda(float(args['xg_lambda']))

    # The following arguments are not directly accessible via the
    # SKLearn API. So whether these work or not, is a bit of a guess.
    sketch_eps = maybe_float(args['xg_sketch_eps'])
    grow_policy = param_xgboost_tree_grow_policy(maybe_int(args['xg_grow_policy']))
    max_leaves = maybe_int(args['xg_max_leaves'])
    normalize_type = param_xgboost_tree_normalize_type(maybe_int(args['xg_normalize_type']))
    rate_drop = maybe_float(args['xg_rate_drop'])
    one_drop = maybe_int(args['xg_one_drop'])
    skip_drop = maybe_float(args['xg_skip_drop'])
    updater = param_xgboost_tree_updater(maybe_int(args['xg_updater']))
    feature_selector = param_xgboost_tree_feature_selector(maybe_int(args['xg_feature_selector']))
    top_k = maybe_int(args['xg_top_k'])

    kwargsd = {
        'sketch_eps': sketch_eps,
        'grow_policy': grow_policy,
        'max_leaves': max_leaves,
        'normalize_type': normalize_type,
        'rate_drop': rate_drop,
        'one_drop': one_drop,
        'skip_drop': skip_drop,
        'updater': updater,
        'feature_selector': feature_selector,
        'top_k': top_k
    }

    # return xgboost.XGBClassifier(objective=objective)
    return xgboost.XGBClassifier(
        objective=objective,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        booster=booster,
        tree_method=tree_method,
        n_jobs=n_jobs,
        gamma=gamma,
        min_child_weight=min_child_weight,
        max_delta_step=max_delta_step,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        colsample_bynode=colsample_bynode,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,

        random_state=random_state,
        verbosity=0,

        kwargs=kwargsd
    )


# `obj` is categorical in [0, 3]
def param_xgboost_objective(obj: int):
    # Problem is multiclass classification!
    # XGBoost should utilize a voting scheme
    # eg. one-vs-one, or one-vs-all
    # for a final classification result otherwise.
    options = [
        'binary:logistic', 
        'binary:logitraw', 
        'binary:hinge',
        'multi:softmax'
        ]
    return options[obj]

# `base_score` is continuous in [0, 1].
#  note: documentation mentions this parameter will have little effect
#        given a large number of iterations.
# def param_xgboost_base_score(base_score: float)
#    return base_score

# `booster` is categorical in [0, 3]
def param_xgboost_booster(booster: int):
    options = [
        'gblinear',
        'dart',
        'gbtree'
        ]
    return options[booster]

# Note: These parameters are listed for both tree (dart, gbtree) and
# linear boosting techniques.

# (parameter) `lambda` is continuous in (0, ∞] (?)
def param_xgboost_lambda(lmbd: float):
    return np.exp(lmbd)

# (parameter) `alpha` is continuous in (0, ∞] (?)
def param_xgboost_alpha(alpha: float):
    return np.exp(alpha) - 0.0001


# The following parameters are utilized only with tree (dart, gbtree)
# boosting techniques.

# (parameter) `eta` is continuous in [0, 1]
# note: via sklearn this parameter is called `learning_rate`
def param_xgboost_learning_rate(eta: Union[float, None]):
    if eta is None:
        return None

    return np.exp(eta) - 0.0001

# (parameter) `gamma` is continuous in [0, ∞]
# def param_xgboost_tree_gamma(gamma: float):
#    return gamma

# (parameter) `min_child_weight` is integer in [0, ∞]
# def param_xgboost_tree_min_child_weight(min_child_weight : int):
#    return min_child_weight

# (parameter) `max_delta_step` is integer in [0, ∞]
#   recommended: default is 0, but with logistic regression and class
#   imbalance, a better value is potentially in [1, 10].
# def param_xgboost_tree_max_delta_step (max_delta_step : int):
#    return max_delta_step

# (unused) `sampling_method` is categorical in [0, 1]
#  note: Is fixed to 0 for anything other than gpu hist...  
# def param_xgboost_tree_sampling_method(sampling_method: int):
#     options = [
#         'uniform',
#         'gradient_based'
#         ]
#     return options[sampling_method]

# Sampling parameters.
# - subsample indicates the size of the fraction sampled from the training data for each boosting iteration
# - colsample_bytree subsamples features per tree
# - colsample_bylevel samples a fraction of features from "colsample_bytree" per layer
# - colsample_bynode samples a fraction of features from "colsample_bylevel" per split node.

# (parameter) `subsample` is continuous in (0, 1]
#  Recommended value: >= 0.5 if sampling method is 'uniform'
#  gradient based can go as low as 0.1 without loss of accuracy.
# def param_xgboost_tree_subsample(subsample: float):
#    return subsample

# (parameter) `colsample_bytree`, (parameter) `colsample_bylevel`, (parameter) `colsample_bynode` 
# are all continuous in (0, 1]
# def param_xgboost_tree_colsample_bytree(colsample_bytree: float):
#    return colsample_bytree
# def param_xgboost_tree_colsample_bylevel(colsample_bylevel: float):
#    return colsample_bylevel
# def param_xgboost_tree_colsample_bynode(colsample_bynode: float):
#    return colsample_bynode

# (parameter) `tree_method` is categorical in [0, 3]
def param_xgboost_tree_tree_method(tree_method: Union[int, None]):
    if tree_method is None:
        return None

    options = [
        'auto', 'exact', 'approx', 'hist'
    ]
    return options[tree_method]

# (parameter) `sketch_eps` is continuous in (0, 1)
# note: only used when `tree_method` = 'approx'
# def param_xgboost_tree_sketch_eps(sketch_eps: float):
#    return sketch_eps

# (parameter) `grow_policy` is categorical in [0, 1]
# note: only used when `tree_method` = 'hist'
def param_xgboost_tree_grow_policy(grow_policy: Union[int, None]):
    if grow_policy is None:
        return None

    options = [
        'depthwise', 'lossguide'
    ]
    return options[grow_policy]

# (parameter) `max_leaves` is integer in [0, ∞]
# note: only relevant if `grow_policy` = 'lossguide'
# def param_xgboost_tree_max_leaves(max_leaves: int):
#    return max_leaves

# (unused-parameter) `sample_type` is categorical in [0, 1]
# def param_xgboost_tree_sample_type(sample_type: int):
#     options = [
#         'uniform', 'weighted'
#     ]
#     return options[sample_type]

# (parameter) `normalize_type` is categorical in [0, 1]
def param_xgboost_tree_normalize_type(normalize_type: Union[int, None]):
    if normalize_type is None:
        return None

    options = [
        'tree', 'forest'
    ]
    return options[normalize_type]

# (parameter) `rate_drop` is continuous in [0, 1]
# def param_xgboost_tree_rate_drop(rate_drop: float):
#    return rate_drop

# (parameter) `one_drop` is categorical in [0, 1]
# def param_xgboost_tree_one_drop(one_drop: int):
#    return one_drop

# (parameter) `skip_drop` is continuous in [0, 1]
# def param_xgboost_tree_skip_drop(skip_drop: float):
#    return skip_drop


# The following parameters are utilized only with gblinear as boosting technique.
# (parameter) `updater` is categorical in [0, 1]
def param_xgboost_tree_updater(updater: Union[int, None]):
    if updater is None:
        return None

    options = [
        'shotgun', 'coord_descent'
    ]
    return options[updater]

# The following parameters are utilized only with gblinear as boosting technique.
# (parameter) `feature_selector` is categorical in [0, 4]
def param_xgboost_tree_feature_selector(feature_selector: Union[int, None]):
    if feature_selector is None:
        return None
    
    options = [
        'cyclic', 'shuffle', 'random', 'greedy', 'thrifty'
    ]
    return options[feature_selector]

# (parameter) `top_k` is integer in [0, ∞]
# note: `top_k` = 0 means select all. 
# def param_xgboost_tree_top_k(top_k: int):
#    return top_k

# (parameter) `num_round` is integer in [0, ∞]
# note: We'll certainly want to limit this, as this is directly related to the number of trees
# as well as the computational time itself.
# def param_xgboost_tree_num_round(num_round: int):
#    return num_round

# Additional important facts:
#  `n_jobs` = 1 - to avoid parallel processing

