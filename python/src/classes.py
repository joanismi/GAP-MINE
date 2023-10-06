import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_recall_curve, make_scorer, confusion_matrix,\
    precision_score, recall_score, matthews_corrcoef, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from classifiers import tuckeys_fences
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
import os

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


def gapmine2(X, y, beta=1, false_annotations=False, shortest_paths=None, graph=None, random_state=None):
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

    
    # scale rwr values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    ########################### Cross-validation #####################################
    clf = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 15),
        scoring=scorer,
        penalty='l1',
        solver='liblinear',
        cv=10,
        n_jobs=None,
        verbose=0,
        refit=True,
        )
    
    if not sys.warnoptions:
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FitFailedWarning)
        os.environ["PYTHONWARNINGS"] = "ignore"
        clf.fit(X_train, y_train)
    
    ######################## calibrate threshold ##############################################

    threshold = db_calibration(X_train, y_train, clf, beta=beta)
    
    ############################### classification results ##############################################
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = db_classifier(y_pred_proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
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
        precision_at_k = top_k_precision(y_test, y_pred_proba),
        random_precision = (tp+fn)/(tp+fp+tn+fn)
    )
    
    return clf_results, clf


def feature_selection(X, y, max_n_components=10, train_size=0.8, variance_cutoff=0.9, random_state=None):
    """
    Computes feature selection using PLS-DA.
    """

    # split the train set to evaluate PLS model
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=random_state)

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
    outliers = np.flatnonzero(coef>=threshold)

    # top half of upper outliers features
    c = coef[coef>=threshold]
    top_outliers = outliers[c >= np.median(c)]
    
    features = (top10, outliers, top_outliers)
    
    return features, n_components, explained_variance


######## feature selection with vip scores and explained variance ratio ###################
def feature_selection(X, y, max_n_components=10, variance_cutoff=0.9):
    """
    Computes feature selection using PLS-DA.
    """

    # scale module rwr
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    plsda = PLSRegression(n_components=max_n_components, scale=False).fit(X, y)

    explained_variance_ratio_ = explained_variance_ratio(plsda, y)
    
    components = np.flatnonzero(np.cumsum(explained_variance_ratio_) > variance_cutoff)
    if components.shape[0] > 0:

        n_components = components[0] + 1
        plsda = PLSRegression(n_components=n_components, scale=False).fit(X, y)
        vips = vip_scores(plsda)

    else:

        n_components = max_n_components
        plsda = PLSRegression(n_components=n_components, scale=False).fit(X, y)
        vips = vip_scores(plsda)

    threshold = tuckeys_fences(vips)
    print(threshold)
    # use 3 distinct feature selection methods
    # 10 most important features
    top10 = np.sort(np.argsort(vips)[-10:])

    # upper outlier features
    outliers = np.flatnonzero(vips>=threshold)
    print(outliers)
    # top half of upper outliers features
    c = vips[vips>=threshold]
    top_outliers = outliers[c >= np.median(c)]

    # VIP > 1
    top_vip = np.flatnonzero(vips>1)
    features = (top10, outliers, top_outliers, top_vip)
    
    return features, n_components, explained_variance_ratio_, vips


def explained_variance_ratio(model, y):

    T = model.x_scores_
    Q = model.y_loadings_
    explained_variance = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    total_var = np.sum((y-np.mean(y))**2)

    return explained_variance/total_var


def vip_scores(model):
    
    T = model.x_scores_
    W = model.x_rotations_
    Q = model.y_loadings_
    w = W.shape[0]
    s = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    w_norm = W / np.linalg.norm(W, axis=0)
    vips = np.sqrt(w * np.sum(s * w_norm ** 2, axis=1) / np.sum(s, axis=0))
    return vips


############################################################################################
def db_classifier(y_pred_proba, threshold, neg_class=0):
    """
    
    """
    
    y_pred = np.ones_like(y_pred_proba)
    y_pred[np.flatnonzero(y_pred_proba<threshold)] = neg_class
    
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


def fbeta_array(precision_scores, recall_scores, beta=1):
    """
    Calculates the f-beta score for all elements in an array. 
    """
    f_beta = (1+beta**2)*precision_scores*recall_scores/((beta**2)*precision_scores+recall_scores)
    return f_beta


def db_optimizer(y_true, y_pred_proba, beta=1, sample_weight=None, neg_class=0, return_threshold=False): #
    """
    """        
    precision_scores, recall_scores, thresholds_ = precision_recall_curve(y_true, y_pred_proba)
    f_beta = fbeta_array(precision_scores, recall_scores, beta=beta)
    threshold = thresholds_[np.nanargmax(f_beta)]

    f1 = f1_scorer(y_true, y_pred_proba, threshold, sample_weight=sample_weight, neg_class=neg_class)
    
    if return_threshold:
        return f1, threshold
    else:
        return f1
    

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
        

