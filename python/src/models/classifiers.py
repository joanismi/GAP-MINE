from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin, clone


import sys
sys.path.append('../features')
from disease_process_proteins import process_selector

from joblib import Parallel, delayed

import os
import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
np.seterr('ignore')

from sklearn.metrics import confusion_matrix, precision_score, fbeta_score, make_scorer,\
    recall_score, f1_score, matthews_corrcoef, precision_recall_curve, auc, roc_auc_score, average_precision_score

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

from igraph import Graph
from joblib import Parallel, delayed


def baseline_classifier(metrics, labels, module_labels, fa=None, n_jobs=-1):
    """
    
    """

    def baseline_clf(X, y, fa=None):
                    
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)

        train = pd.DataFrame({'values': X_train, 'labels': y_train})
        test = pd.DataFrame({'values': X_test, 'labels': y_test})
        
        ############Train model & set threshold####################
        
        train = train.sort_values(by=['values'], ascending=False)
        
        X_train = train['values'].values
        y_train = train['labels'].values
        y_train_positives = sum(y_train)
        y_train_total_labels = len(y_train)

        
        TP = np.cumsum(y_train)
        FP = np.array(range(1, y_train_total_labels+1)) - TP
        FN = y_train_positives - TP
        TN = y_train_total_labels - FP - TP - FN
        
        precision = TP/(TP+FP)

        recall = TP/(TP+FN)

        
        f_measure = 2 * (precision*recall)/(precision+recall)
            
        best_result_index = np.nanargmax(f_measure)
        f_measure = list(f_measure)
        threshold = X_train[best_result_index]
        
        #####################classification results##################################
        if fa is not None:
            fa_proteins = list(set(fa)&set(test.index))
            
            test.loc[test.index.isin(fa_proteins), 'labels'] = 0
        
        
        test = test.sort_values(by=['values'], ascending=False)
        test_positives = test.loc[test['values'] > threshold]['labels']
        tot_pos = int(sum(test['labels']))

        TP = sum(test_positives)
        FP = len(test_positives) - TP
        FN = sum(test['labels'])-TP
        TN = len(test['labels'])-FP-TP-FN
        
        # calculate precision
        if (TP+FP) == 0:
            precision = 0
        else:
            precision = TP/(TP+FP)

        # calculate recall
        if (TP+FN) == 0:
            recall = 0
        else:
            recall = TP/(TP+FN)

        # calculate mcc
        a = TP*TN-FP*FN
        b = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if b==0:
            mcc = 0
        else:
            mcc = a/(b**0.5)
        
        mcc = np.where(np.isinf(mcc), -np.Inf, mcc)

        # Calculate f-measure
        if (precision+recall) == 0:
            f_measure = 0
        else:
            f_measure = 2 * ((precision*recall)/(precision+recall))


        precision_at_20 = sum(test['labels'][:20])/20
        precision_at_15 = sum(test['labels'][:15])/15
        precision_at_20p = sum(test['labels'][:tot_pos])/tot_pos

        precision_scores, recall_scores, thresholds_pr  = precision_recall_curve(test['labels'], test['values'])
        auprc = auc(recall_scores, precision_scores)
        random_precision = (TP+FN)/(TP+FP+TN+FN)
        
        results = {
            'f_measure': f_measure,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
            'tp': TP,
            'fp': FP,
            'fn': FN,
            'tn': TN,
            'precision@15': precision_at_15,
            'precision@20': precision_at_20,
            'precision@k_random': precision_at_20p/random_precision,
            'auprc_random': auprc/random_precision
        }
        return results

    if fa is not None:
        classifier_results = Parallel(n_jobs=n_jobs)(
            delayed(baseline_clf)(
            metrics[module], labels[module], fa.loc[module]) for module in tqdm(module_labels)
            )
    else:
        classifier_results = Parallel(n_jobs=n_jobs)(
            delayed(baseline_clf)(
            metrics[module], labels[module], None) for module in tqdm(module_labels)
            )
    
    classifier_results = pd.DataFrame(classifier_results)
    classifier_results['module'] = module_labels

    return classifier_results


def logistic_classifier(metrics, fs, labels, module_labels, fa=None, n_jobs=-1):
    """
    metrics: df
    fs: df
    labels: df
    fa: df/series
    module_labels: list
    """

    clf_results = []
    cv_results = []
    clf_models = []
    
    for i, module in enumerate(tqdm(module_labels)):
        clf = None
        mean_f_measure = -1
        std_f_measure = -1
        for method in ['10', 'middle', 'outlier10']:
            fs_ = process_selector(fs, i, method)
            X = metrics.iloc[:, fs_]
            y = labels.loc[:, module]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)
            
            clf = LogisticRegression(
                penalty=None,
                solver='lbfgs',
                max_iter=100,
                n_jobs=n_jobs,
                verbose=0
                )
            
            cv_split = StratifiedKFold(n_splits=10)

            if not sys.warnoptions:
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore", category=FitFailedWarning)
                os.environ["PYTHONWARNINGS"] = "ignore"
                cv = cross_val_score(clf, X_train, y_train, scoring='f1', cv=cv_split, n_jobs=n_jobs)
                clf.fit(X_train, y_train)
            
            cv_mean, cv_std = cv.mean(), cv.std()
            
            # choose fs that generates the best model
            if cv_mean > mean_f_measure:
                mean_f_measure = cv_mean
                std_f_measure = cv_std
                best_clf = clf
                X_test_best = X_test
                X_train_best = X_train
                y_test_best = y_test
                n_predictors = fs_.shape[0]
                cv_results = method
                fs_indices = fs_
            
            # if cv results are identical, choose the model 
            # with fewer predictors
            
            elif (cv_mean == mean_f_measure) & (fs_.shape[0] < n_predictors):
                mean_f_measure = cv_mean
                std_f_measure = cv_std
                best_clf = clf
                X_test_best = X_test
                X_train_best = X_train
                y_test_best = y_test
                n_predictors = fs_.shape[0]
                cv_results = method
                fs_indices = fs_

        cv_results.append((module, cv_results, mean_f_measure, std_f_measure, n_predictors, fs_indices))
        clf_models.append(best_clf)
        
        y_train_pred = best_clf.predict_proba(X_train_best)[:, 1]
        y_pred_proba = best_clf.predict_proba(X_test_best)[:, 1]

        ########################calibrate threshold##############################################
        
        value_labels = pd.DataFrame({'value': y_train_pred, 'label': y_train})
        
        value_labels.sort_values(by=['value'], ascending=False, inplace=True)
        
        tot_pos = sum(value_labels['label']) # total number of positives
        ord_labels = np.array(value_labels['label'].values)
        tot_labels = len(ord_labels)
        TP = np.cumsum(ord_labels)
        FP = np.arange(1, tot_labels+1) - TP
        FN = tot_pos-TP
        TN = tot_labels-FP-TP-FN
    
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            f_measure = 2*((precision*recall)/(precision+recall))

        best_result_index = np.nanargmax(f_measure)
        threshold = value_labels['value'].values[best_result_index]
        
        ###############################classification results##############################################
        value_labels = pd.DataFrame({'value': y_pred_proba, 'label': y_test_best})
        
        # Correct false annotations and compute classifier metrics
        if fa is not None:
            fa_proteins = list(set(value_labels.index) & set(fa.loc[module]))
            
            for fa_protein in fa_proteins:
                value_labels.loc[value_labels.index==fa_protein, 'label'] = 0

        value_labels.sort_values(by=['value'], ascending=False, inplace=True)
        tot_pos = sum(value_labels['label'])
        tot_labels = len(value_labels['label'])
        TP = sum(value_labels[value_labels['value'] >= threshold]['label'])
        FP = len(value_labels[value_labels['value'] >= threshold]['label']) - TP
        FN = tot_pos-TP
        TN = tot_labels-FP-TP-FN
    
        precision_at_20p = sum(value_labels['label'][:int(tot_pos)])/int(tot_pos)
        precision_scores, recall_scores, thresholds_pr  = precision_recall_curve(value_labels['label'], value_labels['value'])
        auprc = auc(recall_scores, precision_scores)
        random_precision = (TP+FN)/(TP+FP+FN+TN)

        # calculate mcc
        a = TP*TN-FP*FN
        b = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if b==0:
            mcc = 0
        else:
            mcc = a/(b**0.5)
            
        # calculate precision
        if (TP+FP) == 0:
            precision = 0
        else:
            precision = TP/(TP+FP)
            
        # calculate recall
        if (TP+FN) == 0:
            recall = 0
        else:
            recall = TP/(TP+FN)
        
        # Calculate f-measure
        if (precision+recall) == 0:
            f_measure = 0
        else:
            f_measure = 2*((precision*recall)/(precision+recall))           
        
        clf_results.append({
            'module': module,
            'f_measure': f_measure,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
            'tp': TP,
            'fp': FP,
            'fn': FN,
            'tn': TN,
            'precision@k_random': precision_at_20p/random_precision,
            'auprc_random': auprc/random_precision,
            'threshold': threshold
        })
    
    clf_results = pd.DataFrame(clf_results)
    cv_results = pd.DataFrame(cv_results, columns=['module', 'fs_method', 'mean_f_measure', 'std_f_measure', 'n_predictors', 'predictor_indices'])

    return clf_results, cv_results, clf_models


def whole_clf(data, labels, models, fs, n_fs, threshold):
    # Predicts whether a protein is associated with a process/disease in the whole dataset using the previously trained model.
    #
    # INPUT:
    #   - dataframe with metrics.
    #   - dataframe with labels
    #   - list with trained models.
    #   - dataframe with VIP scores.
    #   - list with number of selected modules.
    #   - list with probability thresholds.
    #
    # RETURNS: dataframe with a collection of performance metrics.

    clf_results = []
    for i in tqdm(range(data.shape[0])):
        module_fs = fs.iloc[:, i]
        y_true = labels.iloc[:, i]
        module_data = data[list(module_fs.sort_values(
            ascending=False)[:n_fs[i]].index)].transpose()
        module_threshold = threshold[i]
        module_model = models[i]
        module_model.fit(module_data, y_true.values)
        y_pred = module_model.predict_proba(module_data)[:, 1]
        y_pred[y_pred < module_threshold] = 0
        y_pred[y_pred >= module_threshold] = 1
        true_pred_df = pd.DataFrame({'true': y_true.values, 'pred': y_pred})
        true_pred_df.index = y_true.index
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()


        clf_results.append({
                    #'module': module,
                    'f_measure': f1_score(y_true, y_pred, zero_division=0),
                    'precision': precision_score(y_true, y_pred),
                    'recall': recall_score(y_true, y_pred),
                    'mcc': matthews_corrcoef(y_true, y_pred),
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn,
                    'new_proteins': list(
            true_pred_df[(true_pred_df['true'] == 0) & (true_pred_df['pred'] == 1)].index
            ),
            })

        
    return pd.DataFrame(clf_results)


########################################### NEW ##########################################################
def tuckeys_fences(s, fence='upper', k=1.5):
    """
    Computes the upper or lower Tukey's fences. This method is commonly
    used to detect outliers and to define the whiskers in box plots.
    
    Parameters
    ----------
    s : numpy 1D array or Series
        Sample where tuckey's fences method is applied.
        
    fence : {'upper', 'lower'}, default 'upper'
        Type of fence to compute.
        
    k : int or float, default 1.5
        k determines the reach of the fence. The higher is k, the more
        strigent the detection method is. 
        k = 1.5 usually defines the inner fence and k = 3 defines the
        outer fence.
        
    Returns
    -------
    fence_val : Tuckey's fence value

    """
    
    assert fence in ['upper', 'lower'], "not a tuckey's fence"

    IQR = np.quantile(s, .75) - np.quantile(s, .25)

    if fence == 'lower':
        fence_val = np.quantile(s, .25) - k*IQR
        
    else:
        fence_val = np.quantile(s, .75) + k*IQR
    
    return fence_val

# Criar Classe FS
def feature_selection(X, y, max_n_components=10, train_size=0.8, variance_cutoff=0.9):
    """
    Computes feature selection using PLS-DA.
    """

    # split the train set to evaluate PLS model
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)

    # scale module rwr
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    for n_components in range(1, max_n_components+1):

        plsda = PLSRegression(n_components=n_components, scale=False).fit(X_train, y_train)
        
        # evaluate model using R²
        explained_variance = plsda.score(X_test, y_test)

        if explained_variance >= variance_cutoff:            
            break
    
    # use model coefficient to evaluate feature importance
    if not sys.warnoptions:
        warnings.simplefilter("ignore", category=FutureWarning)
        coef = np.abs(plsda.coef_.flatten())
    threshold = tuckeys_fences(coef)
    
    # use 3 distinct feature selection methods
    # 10 most important features
    top10 = np.argsort(coef)[-10:]

    # upper outlier features
    indices = np.arange(X.shape[1])
    outliers = indices[coef>=threshold]

    # top half of upper outliers features
    i = indices[coef>=threshold]
    
    c = coef[coef>=threshold]
    
    top_outliers = i[c >= np.median(c)]
    
    features = (top10, outliers, top_outliers)
    
    return features, n_components, explained_variance


def add_false_annotations(y, graph, sp, random_state=None):
    """
    Adds false annotations to a module. Selects proteins with no known association 
    to a module and annotates them as associated. Proteins closer to module proteins have a
    higher probability of being wrongly annotated.

    Parameters:
    ---------
    y: array-like 1D
        
    graph: array-like 2D

    sp: array-like 2D
    
    random_state: int, RandomState instance or None, default=None

    Returns:
    --------
    y_fa: 1D array
        Module labels with false annotations.
    """
    rng = np.random.default_rng(random_state)
    y_fa = y.copy()

    
    module_proteins = np.arange(y_fa.shape[0])[y_fa == 1]
    other_proteins = np.arange(y_fa.shape[0])[y_fa == 0]

    min_sp = np.min(sp[np.ix_(other_proteins, module_proteins)], axis=1)

    degree_values = np.log10(graph.degree(other_proteins))
    weight = degree_values/10**min_sp
    normalized_weight = weight/np.sum(weight)

    fa_proteins = rng.choice(other_proteins, int(module_proteins.shape[0]*0.1), p=normalized_weight)
    y_fa[fa_proteins] = 1
    
    return y_fa


def gapmine(X, y, beta=0.5, false_annotations=False, shortest_paths=None, graph=None, random_state=None):
    """
    
    """
    
    scorer = make_scorer(fbeta_score, beta=beta, needs_proba=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=random_state
        )
    
    if false_annotations:
        if shortest_paths is None:
            raise ValueError('Array with shortest paths needed to compute false annotations.')
        elif graph is None:
            raise ValueError('Graph is needed to compute false annotations.')
        else:
            
            y_train = add_false_annotations(y_train, graph, shortest_paths, random_state=random_state)

    # feature selection
    features, _, _ = feature_selection(
        X_train, y_train, max_n_components=10, train_size=0.8, variance_cutoff=0.9
        )
    
    # scale rwr values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    cv_results = {'best_score': 0, 'best_C': 0, 'n_iter': 0, 'n_features': 0, 'fs': 0}
    for fs in features:
        
        n_features = len(fs)

        X_train_fs = X_train[:, fs]
        X_test_fs = X_test[:, fs]
    
        clf = LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 15),
            scoring=scorer,
            penalty='l2',
            solver='lbfgs',
            cv=10,
            n_jobs=None,
            verbose=0,
            refit=True,
            )
        
        if not sys.warnoptions:
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=FitFailedWarning)
            os.environ["PYTHONWARNINGS"] = "ignore"
            clf.fit(X_train_fs, y_train)

        mean_scores = np.mean(clf.scores_[1.0], axis=0)
        mean_iter = np.mean(clf.n_iter_[0], axis=0)
        
        best_index = np.argmax(mean_scores)
        best_score = np.amax(mean_scores)
        best_C = clf.C_[0]
        n_iter = mean_iter[best_index]
                    
        # choose fs that generates the best model
        if best_score > cv_results['best_score']:

            cv_results['best_score'] = best_score
            cv_results['best_C'] = best_C
            cv_results['n_iter'] = n_iter
            cv_results['n_features'] = n_features
            cv_results['fs'] = fs
            best_clf = clf
            X_train_best = X_train_fs
            X_test_best = X_test_fs

        # if cv results are identical, choose the model 
        # with fewer predictors
        elif (best_score == cv_results['best_score']) & (n_features < cv_results['n_features']):
            
            cv_results['best_score'] = best_score
            cv_results['best_C'] = best_C
            cv_results['n_iter'] = n_iter
            cv_results['n_features'] = n_features
            cv_results['fs'] = fs
            best_clf = clf
            X_train_best = X_train_fs
            X_test_best = X_test_fs


    ######################## calibrate threshold ##############################################

    skf = StratifiedKFold(n_splits=10)
    split = skf.split(X_train_best, y_train)

    thresholds = []
    for train_index, test_index in split:
        X_train_thr,  y_train_thr = X_train_best[train_index], y_train[train_index]
        X_test_thr, y_test_thr = X_train_best[test_index], y_train[test_index]
        
        log_reg = LogisticRegression(C=best_clf.C_[0])
        log_reg.fit(X_train_thr, y_train_thr)
        y_pred_proba = log_reg.predict_proba(X_test_thr)[:, 1]
        
        precision_scores, recall_scores, thresholds_ = precision_recall_curve(y_test_thr, y_pred_proba)
        f_beta = (1+beta**2)*precision_scores*recall_scores/((beta**2)*precision_scores+recall_scores)
        threshold = thresholds_[np.nanargmax(f_beta)]
                
        thresholds.append(threshold)
    
    threshold = np.mean(thresholds)
    
    ############################### classification results ##############################################
    
    y_pred_proba = best_clf.predict_proba(X_test_best)[:, 1]
    y_pred = db_classifier(y_pred_proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    precision_scores, recall_scores, _ = precision_recall_curve(y_test, y_pred_proba)
    precision_at_k = top_k_precision(y_test, y_pred_proba)
    
    clf_results = dict(
        threshold = threshold,
        tn = tn, fp = fp, fn = fn, tp = tp,
        precision = precision_score(y_test, y_pred, zero_division=0),
        recall = recall_score(y_test, y_pred, zero_division=0),
        f_beta = fbeta_score(y_test, y_pred, beta=beta),
        f1 = fbeta_score(y_test, y_pred, beta=1),
        phi_coef = matthews_corrcoef(y_test, y_pred),
        p4 = 4*tp*tn/(4*tp*tn+(tp+tn)*(fp+fn)),
        auprc = auc(recall_scores, precision_scores),
        precision_at_k = precision_at_k,
        precision_at_k_random = precision_at_k/((tp+fn)/(tp+fp+tn+fn))
    )
    
    return cv_results, clf_results, best_clf


def baseline(X, y, beta=0.5, false_annotations=False, shortest_paths=None, graph=None, random_state=None):
    """
    
    """
                    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=random_state)
    
    if false_annotations:

        if shortest_paths is None:

            raise ValueError('Array with shortest paths needed to compute false annotations.')
        elif graph is None:

            raise ValueError('Graph is needed to compute false annotations.')
        else:
            
            y_train = add_false_annotations(y_train, graph, shortest_paths, random_state=random_state)

    ############ Train model & set threshold ####################
    
    precision_scores, recall_scores, thresholds_ = precision_recall_curve(y_train, X_train)
    f_beta = (1+beta**2)*precision_scores*recall_scores/((beta**2)*precision_scores+recall_scores)
    threshold = thresholds_[np.nanargmax(f_beta)]
    
    ############
    y_pred = db_classifier(X_test, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    precision_scores, recall_scores, _ = precision_recall_curve(y_test, X_test)
    precision_at_k = top_k_precision(y_test, X_test)

    clf_results = dict(
        threshold = threshold,
        tn = tn, fp = fp, fn = fn, tp = tp,
        precision = precision_score(y_test, y_pred, zero_division=0),
        recall = recall_score(y_test, y_pred, zero_division=0),
        f_beta = fbeta_score(y_test, y_pred, beta=beta),
        f1 = fbeta_score(y_test, y_pred, beta=1),
        phi_coef = matthews_corrcoef(y_test, y_pred),
        p4 = 4*tp*tn/(4*tp*tn+(tp+tn)*(fp+fn)),
        auprc = auc(recall_scores, precision_scores),
        precision_at_k = precision_at_k,
        precision_at_k_random = precision_at_k/((tp+fn)/(tp+fp+tn+fn))
    )

    return clf_results

#####################################################################################################################################


def top_k_precision(y_true, y_pred_proba):
    """
    Calculates the precision at (P@K) where K is the number of positives
    in the test set.
    """
    k = np.sum(y_true) # n_positives
    
    top_k = y_true[np.argsort(y_pred_proba)[::-1]][:k]

    # TP/(TP+FP)
    precision = np.sum(top_k)/k

    return precision


def db_classifier(y_pred_proba, threshold, neg_class=0):
    """
    
    """
    indexer = np.arange(y_pred_proba.shape[0])
    y_pred = np.ones_like(y_pred_proba)
    y_pred[indexer[y_pred_proba<threshold]] = neg_class
    return y_pred


def f1_scorer(y_true, y_pred_proba, threshold, sample_weight=None, neg_class=0):
    """
    
    """
    y_pred = db_classifier(y_pred_proba, threshold, neg_class=neg_class)
    
    f1 = f1_score(
        y_true, y_pred, labels=None, pos_label=1, average='binary', 
        sample_weight=sample_weight, zero_division=0
        )
    
    return f1


def db_optimizer(y_true, y_pred_proba, beta=1, sample_weight=None, neg_class=0, return_threshold=False): #
    """
    """        
    precision_scores, recall_scores, thresholds_ = precision_recall_curve(y_true, y_pred_proba)
    f_beta = (1+beta**2)*precision_scores*recall_scores/((beta**2)*precision_scores+recall_scores)
    threshold = thresholds_[np.nanargmax(f_beta)]

    f1 = f1_scorer(y_true, y_pred_proba, threshold, sample_weight=sample_weight, neg_class=neg_class)
    
    if return_threshold:
        return f1, threshold
    else:
        return f1


def calibrate_db(estimator, X, y, cv, sample_weight=None):

        ############# OPTIMIZE DB ####################
        
        if isinstance(cv, int):
            skf = StratifiedKFold(n_splits=cv)
        else:
            skf = cv
        
        splits = skf.split(X, y)

        thresholds = list()
        for train_index, test_index in splits:
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            
            log_reg = clone(estimator)
            
            log_reg.fit(X_train, y_train)
            
            y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
            
            _, threshold = db_optimizer(
                y_test,
                y_pred_proba,
                sample_weight=sample_weight,
                neg_class=0,
                return_threshold=True
                )
            
            thresholds.append(threshold)    
                
        
        threshold = np.mean(thresholds)
        
        return threshold


class LogRegClassifier(LogisticRegressionCV, ClassifierMixin):
    """
    Logistic Regression CV classifier with C parameter and decision boundary tunning.
    """
    
    def __init__(
        self,
        *,
        Cs=10,
        cv=None,
        scoring='f1',
        tol=1e-4,
        max_iter=100,
        class_weight=None,
        n_jobs=None,
        random_state=None,
    ):
        self.Cs = Cs
        self.fit_intercept = True
        self.cv = cv
        self.dual = False
        self.penalty = "l2"
        self.tol = tol
        self.scoring = scoring
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = 0
        self.solver = "lbfgs"
        self.refit = True
        self.intercept_scaling = 1.0
        self.multi_class = 'auto' # binary problem
        self.random_state = random_state    
        self.l1_ratios = None


    def fit(self, X, y, sample_weight=None):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self
            Fitted estimator.
        """
                
        #self.scoring = make_scorer(
            #db_optimizer,
            #sample_weight=sample_weight,
            #neg_class=-1,
            #return_threshold=False,
            #needs_proba=True,
            #greater_is_better=True
            #)
        
        ############# FIT #########################
        super().fit(X, y, sample_weight=sample_weight)

        return self
    

    def calibrate_db(self, X, y, sample_weight=None):

        ############# OPTIMIZE DB ####################
        cv = self.cv
        if isinstance(cv, int):
            skf = StratifiedKFold(n_splits=cv)
        else:
            skf = cv
        
        splits = skf.split(X, y)

        thresholds = list()
        for train_index, test_index in splits:
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            
            log_reg = LogisticRegression(
                C=self.C_[0],
                fit_intercept=self.fit_intercept,
                dual=self.dual,
                penalty=self.penalty,
                tol=self.tol,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                solver=self.solver,
                intercept_scaling=self.intercept_scaling,
                multi_class=self.multi_class,
                random_state=self.random_state,
                l1_ratio=self.l1_ratios,
                )
            
            log_reg.fit(X_train, y_train)
            
            y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
            
            _, threshold = db_optimizer(
                y_test,
                y_pred_proba,
                sample_weight=sample_weight,
                neg_class=0,
                return_threshold=True
                )
            
            thresholds.append(threshold)    
                
        self.thresholds_ = thresholds
        self.threshold_ = np.mean(thresholds)

        return self


    def score(self, X, y, sample_weight=None):
        """Score using the `scoring` option on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score of self.predict(X) w.r.t. y.
        """
        y_pred = self.predict_proba(X)[:, 1]
        score = f1_scorer(
            y,
            y_pred,
            threshold=self.threshold_,
            neg_class=0,
            sample_weight=sample_weight
            )
        return score


    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        y_pred_proba = self.predict_proba(X)[:,1]

        y_pred = db_classifier(y_pred_proba, self.threshold_)

        return y_pred
        

