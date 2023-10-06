from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd


import os, sys, warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
np.seterr('ignore')

from sklearn.metrics import confusion_matrix, precision_score, fbeta_score, make_scorer,\
    recall_score, f1_score, matthews_corrcoef, precision_recall_curve, auc, average_precision_score

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

from igraph import Graph

from rpy2.robjects import r, numpy2ri, default_converter, IntVector, FactorVector
from rpy2.robjects.packages import importr
import rpy2

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

#import rpy2.rinterface_lib.callbacks rpy2.rinterface_lib.callbacks.consolewrite_warnerror = my_callback, my_callback = lambda *args: None
def feature_selection(X, y, module, cv=5):
    """
    Computes feature selection using OPLS-DA.
    """
    # suppress outputs to R console
    buf = []
    def warn(x):
        buf.append(x)

    consolewrite_warnerror_backup = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
    rpy2.rinterface_lib.callbacks.consolewrite_warnerror = warn

    ropls = importr('ropls')
    # Create a converter that starts with rpy2's default converter
    # to which the numpy conversion rules are added.
    np_cv_rules = default_converter + numpy2ri.converter

    with np_cv_rules.context():
                
        oplsda = ropls.opls(
            X,
            FactorVector(IntVector(y)),
            orthoI=r('NA'),
            predI=1,
            crossvalI=cv,
            subset=r('NULL'),
            fig_pdfC="none",
            info_txtC="none"
            )
        vips = ropls.getVipVn(oplsda)

    threshold = tuckeys_fences(vips)
    
    # use 3 distinct feature selection methods
    # 10 most important features
    top10 = np.sort(np.argsort(vips)[-10:])

    # upper outlier features
    outliers = np.flatnonzero(vips>=threshold)
    if outliers.shape[0] > 0:
        
        # top half of upper outliers features
        c = vips[vips>=threshold]
        top_outliers = outliers[c >= np.median(c)]
    
        features = [top10, outliers, top_outliers]
    else:
        features = [top10]
        print(module)
        
    for f in range(len(features)):
        if module not in features[f]:
            features[f] = np.concatenate((features[f], [5]))

    # restore default function
    rpy2.rinterface_lib.callbacks.consolewrite_warnerror = consolewrite_warnerror_backup
    
    return features, vips, buf


def add_false_annotations(y, graph, sp, added_pct=0.1, random_state=None):
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

    module_proteins = np.flatnonzero(y_fa == 1)
    other_proteins = np.logical_not(module_proteins)

    min_sp = np.min(sp[np.ix_(other_proteins, module_proteins)], axis=1)

    degree_values = np.log10(graph.degree(other_proteins))
    weight = degree_values/10**min_sp
    normalized_weight = weight/np.sum(weight)

    fa_proteins = rng.choice(other_proteins, int(module_proteins.shape[0]*added_pct), p=normalized_weight)
    y_fa[fa_proteins] = 1
    
    return y_fa


def gapmine(X, y, module, train_size=0.8, beta=1, false_annotations=False, shortest_paths=None, graph=None, random_state=None):
    """
    
    """
    y = y[:, module]
    scorer = make_scorer(fbeta_score, beta=beta, needs_proba=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
        )
    
    if false_annotations:

        if shortest_paths is None:
            raise ValueError('Array with shortest paths needed to compute false annotations.')
        
        elif graph is None:
            raise ValueError('Graph is needed to compute false annotations.')
        
        else:
            y_train = add_false_annotations(y_train, graph, shortest_paths, random_state=random_state)

    
    # scale rwr values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    # feature selection
    features, _, _ = feature_selection(X_train, y_train, module, cv=5)
    
    ########################### Cross-validation #####################################
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

    threshold = db_calibration(X_train_best, y_train, best_clf, beta=beta)
    
    ############################### classification results ##############################################
    
    y_pred_proba = best_clf.predict_proba(X_test_best)[:, 1]
    y_pred = db_classifier(y_pred_proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

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
        avg_precision = average_precision_score(y_test, y_pred),
        precision_at_k = precision_at_k,
        random_precision = (tp+fn)/(tp+fp+tn+fn)
    )
    
    return cv_results, clf_results, best_clf


def baseline(X, y, beta=1, false_annotations=False, shortest_paths=None, graph=None, random_state=None):
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
    
    threshold = compute_db(y_train, X_train, beta=beta)

    ############ Metrics ################################
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
        random_precision = (tp+fn)/(tp+fp+tn+fn)
    )

    return clf_results


#####################################################################################################################################
def compute_db(y_test, y_pred_proba, beta=1):
    """
    Calculates the decision boundary that maximizes the f_beta score using the precision-recall
    curve.
    """
    precision, recall, thresholds_ = precision_recall_curve(y_test, y_pred_proba)
    
    f_beta = (1+beta**2)*precision*recall/((beta**2)*precision+recall)
    threshold = thresholds_[np.nanargmax(f_beta)]
    
    return threshold
    
    
def db_calibration(X, y, clf, beta=1):
    """
    Calibrates the decision boundary of a logistic regression classifier using cross-validation.
    """
    skf = StratifiedKFold(n_splits=10)
    split = skf.split(X, y)

    thresholds = []
    for train_index, test_index in split:
        X_train,  y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        
        log_reg = LogisticRegression(C=clf.C_[0])
        log_reg.fit(X_train, y_train)
        y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
        
        threshold = compute_db(y_test, y_pred_proba, beta=beta)
                
        thresholds.append(threshold)

    return np.mean(thresholds)


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
    
    y_pred = np.ones_like(y_pred_proba)
    y_pred[np.flatnonzero(y_pred_proba<threshold)] = neg_class
    
    return y_pred


##############################TO MODIFY##############################################################################
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

