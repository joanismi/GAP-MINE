import sys
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score, f1_score, matthews_corrcoef, precision_recall_curve, auc, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from tqdm.notebook import tqdm, tnrange
import numpy as np
import pandas as pd
from statistics import mean, median, stdev
sys.path.append('../features')
from disease_process_proteins import process_selector

import os
import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning


def threshold_classifier(df_values, df_labels, test_indices=False, op_metric='f_measure'):
    # Threshold Classifier for protein-process and protein-disease association prediction by ranking proteins given their scores.
    # Chooses best threshold by maximizing F-Measure (or other metric of choice) during a 10-fold cross validation step.
    #
    # INPUT:
    #   - dataframe with metrics.
    #   - dataframe with protein labels.
    #   - test indices (for direct comparison with logistic classifier).
    #
    # RETURNS: dataframe with a collection of evaluation metrics.

    classifier_results = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'precision': [], 'recall': [
    ], 'f_measure': [], 'mcc': [], 'precision@15': [], 'precision@20': [], 'precision@k_random': [], 'auprc_random': [],}

    for i in tqdm(range(df_values.shape[1])):
    
        process = df_values.columns[i]
        value_labels = pd.concat([pd.DataFrame(df_values[process]), pd.DataFrame(
            df_labels[process])], axis=1, names=['values', 'labels'])
        value_labels.columns = ['values', 'labels']
        test_indices = test_indices

        test_indices_ = test_indices[i][test_indices[i] < 20000]
        train_indices_ = list(
            set(range(0, len(value_labels))) - set(test_indices_))
        train = value_labels.iloc[train_indices_]
        test = value_labels.iloc[test_indices_]
        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(train['values'], train['labels'])
        thresholds = []
        f_measure_train_scores = []
        f_measure_val_scores = []
        for train_index, validation_index in skf.split(train['values'], train['labels']):
            cv_train = train.iloc[train_index].sort_values(
                by=['values'], ascending=False)
            cv_val = train.iloc[validation_index].sort_values(
                by=['values'], ascending=False)
            X_train = cv_train['values'].values
            y_train = cv_train['labels'].values
            y_train_positives = sum(y_train)
            y_train_total_labels = len(y_train)
            TP_train = np.cumsum(y_train)
            FP_train = np.array(range(1, y_train_total_labels+1)) - TP_train
            FN_train = y_train_positives - TP_train
            TN_train = y_train_total_labels-FP_train-TP_train-FN_train
            precision_train = TP_train/(TP_train+FP_train)
            recall_train = TP_train/(TP_train+FN_train)
            f_measure_train = 2 * \
                ((precision_train*recall_train)/(precision_train+recall_train))
            best_result_index = np.nanargmax(f_measure_train)
            f_measure_train = list(f_measure_train)
            threshold = X_train[best_result_index]
            thresholds.append(threshold)
            f_measure_train_scores.append(f_measure_train[best_result_index])
            cv_val_positives = cv_val[cv_val['values'] >= threshold]
            X_val = cv_val['values'].values
            y_val = cv_val['labels'].values
            y_val_positives = sum(cv_val_positives['labels'])
            y_val_total_labels = len(y_val)
            if y_val_total_labels > 0 and y_val_positives > 0:
                TP_val = y_val_positives
                FP_val = len(cv_val_positives) - TP_val
                FN_val = sum(y_val) - TP_val
                TN_val = y_val_total_labels - TP_val - FP_val - FN_val
                precision_val = TP_val/(TP_val+FP_val)
                recall_val = TP_val/(TP_val+FN_val)
                f_measure_val = 2*((precision_val*recall_val) /
                                   (precision_val+recall_val))
                f_measure_val_scores.append(f_measure_val)
            else:
                f_measure_val_scores.append(0)
        test = test.sort_values(by=['values'], ascending=False)
        test_positives = test.loc[test['values'] > mean(thresholds)]['labels']
        TP = sum(test_positives)
        FP = len(test_positives) - TP
        FN = sum(test['labels'])-TP
        TN = len(test['labels'])-FP-TP-FN
        tot_pos = int(sum(test['labels']))
        if TP + FP > 0:
            precision = TP/(TP+FP)
        else:
            precision = 0
        if TP + FN > 0:
            recall = TP/(TP+FN)
        else:
            recall = 0
        accuracy = (TP+TN)/(len(test['labels']))

        precision_at_20 = sum(test['labels'][:20])/20
        precision_at_15 = sum(test['labels'][:15])/15
        precision_at_20p = sum(test['labels'][:tot_pos])/tot_pos
        

        precision_scores, recall_scores, thresholds_pr  = precision_recall_curve(test['labels'], test['values'])
        auprc = auc(recall_scores, precision_scores)
        random_precision = (TP+FN)/(TP+FP+TN+FN)
        try:
            mcc = (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
        except ZeroDivisionError:
            mcc = 0
        try:
            f_measure = 2*((precision*recall)/(precision+recall))
        except ZeroDivisionError:
            f_measure = 0
        mcc = np.where(np.isinf(mcc), -np.Inf, mcc)

        classifier_results['f_measure'].append(f_measure)
        classifier_results['precision'].append(precision)
        classifier_results['recall'].append(recall)
        classifier_results['mcc'].append(mcc)
        classifier_results['tp'].append(TP)
        classifier_results['fp'].append(FP)
        classifier_results['fn'].append(FN)
        classifier_results['tn'].append(TN)
        classifier_results['precision@15'].append(precision_at_15)
        classifier_results['precision@20'].append(precision_at_20)
        classifier_results['precision@k_random'].append(precision_at_20p/random_precision)
        classifier_results['auprc_random'].append(auprc/random_precision)
    results = pd.DataFrame(classifier_results)
    return results


def reduced_classifier_threshold(data_, test_indices_, labels):
    # Threshold Classifier for protein-process and protein-disease association prediction by ranking proteins given their scores.
    # Chooses best threshold by maximizing F-Measure (or other metric of choice) during a 10-fold cross validation step.
    # Given that false positives are included in the positive set, results are adjusted to remove the false positives from the TP and FN values.
    #
    # INPUT:
    #   - list of dataframes with metrics.
    #   - list of labels for each reduction.
    #   - list of test indices (for direct comparison with logistic classifier).
    #
    # RETURNS: dataframe with a collection of evaluation metrics for each metric.

    clf_dfs = []
    for red in tqdm(range(len(data_))):
        data = data_[red]
        labels_ = labels[red]
        test_indices = test_indices_[red]
        clf = threshold_classifier(data, labels_, test_indices)
        clf_dfs.append(clf)
    clf_dfs = pd.concat(clf_dfs)
    return clf_dfs


def threshold_classifier_fp(df_values, df_labels, test_indices=False, op_metric='f_measure'):
    # Threshold Classifier for protein-process and protein-disease association prediction by ranking proteins given their scores.
    # Chooses best threshold by maximizing F-Measure (or other metric of choice) during a 10-fold cross validation step.
    # Given that false positives are included in the positive set, results are adjusted to remove the false positives from the TP and FN values.
    #
    # INPUT:
    #   - dataframe with metrics.
    #   - dataframe with protein labels.
    #   - test indices (for direct comparison with logistic classifier).
    #
    # RETURNS: dataframe with a collection of evaluation metrics.

    classifier_results = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'precision': [], 'recall': [], 'f_measure': [
    ], 'mcc': [], 'precision@15': [], 'precision@20': [], 'precision@n_positives': [], 'tp_proteins': [], 'fp_proteins': [], 'fn_proteins': []}

    for i in tqdm(range(df_values.shape[1])):
        process = df_values.columns[i]
        value_labels = pd.concat([pd.DataFrame(df_values[process]), pd.DataFrame(
            df_labels[process])], axis=1, names=['values', 'labels'])
        value_labels.columns = ['values', 'labels']
        test_indices = test_indices

        test_indices_ = test_indices[i][test_indices[i] < 18000]
        train_indices_ = list(
            set(range(0, len(value_labels))) - set(test_indices_))
        train = value_labels.iloc[train_indices_]
        test = value_labels.iloc[test_indices_]
        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(train['values'], train['labels'])
        thresholds = []
        f_measure_train_scores = []
        f_measure_val_scores = []
        for train_index, validation_index in skf.split(train['values'], train['labels']):
            cv_train = train.iloc[train_index].sort_values(
                by=['values'], ascending=False)
            cv_val = train.iloc[validation_index].sort_values(
                by=['values'], ascending=False)
            X_train = cv_train['values'].values
            y_train = cv_train['labels'].values
            y_train_positives = sum(y_train)
            y_train_total_labels = len(y_train)
            TP_train = np.cumsum(y_train)
            FP_train = np.array(range(1, y_train_total_labels+1)) - TP_train
            FN_train = y_train_positives - TP_train
            TN_train = y_train_total_labels-FP_train-TP_train-FN_train
            precision_train = TP_train/(TP_train+FP_train)
            recall_train = TP_train/(TP_train+FN_train)
            f_measure_train = 2 * \
                ((precision_train*recall_train)/(precision_train+recall_train))
            best_result_index = np.nanargmax(f_measure_train)
            f_measure_train = list(f_measure_train)
            threshold = X_train[best_result_index]
            thresholds.append(threshold)
            f_measure_train_scores.append(f_measure_train[best_result_index])
            cv_val_positives = cv_val[cv_val['values'] >= threshold]
            X_val = cv_val['values'].values
            y_val = cv_val['labels'].values
            y_val_positives = sum(cv_val_positives['labels'])
            y_val_total_labels = len(y_val)
            if y_val_total_labels > 0 and y_val_positives > 0:
                TP_val = y_val_positives
                FP_val = len(cv_val_positives) - TP_val
                FN_val = sum(y_val) - TP_val
                TN_val = y_val_total_labels - TP_val - FP_val - FN_val
                precision_val = TP_val/(TP_val+FP_val)
                recall_val = TP_val/(TP_val+FN_val)
                f_measure_val = 2*((precision_val*recall_val) /
                                   (precision_val+recall_val))
                f_measure_val_scores.append(f_measure_val)
            else:
                f_measure_val_scores.append(0)
        test = test.sort_values(by=['values'], ascending=False)
        test_positives = test.loc[test['values'] >= mean(thresholds)]['labels']
        TP = sum(test_positives)
        FP = len(test_positives) - TP
        FN = sum(test['labels'])-TP
        TN = len(test['labels'])-FP-TP-FN
        tot_pos = int(sum(test['labels']))
        if TP + FP > 0:
            precision = TP/(TP+FP)
        else:
            precision = 0
        recall = TP/(TP+FN)
        accuracy = (TP+TN)/(len(test['labels']))

        precision_at_20 = sum(test['labels'][:20])/20
        precision_at_15 = sum(test['labels'][:15])/15
        precision_at_20p = sum(test['labels'][:tot_pos])/tot_pos
        try:
            mcc = (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
        except ZeroDivisionError:
            mcc = 0
        try:
            f_measure = 2*((precision*recall)/(precision+recall))
        except ZeroDivisionError:
            f_measure = 0
        mcc = np.where(np.isinf(mcc), -np.Inf, mcc)
        classifier_results['f_measure'].append(f_measure)
        classifier_results['precision'].append(precision)
        classifier_results['recall'].append(recall)
        classifier_results['mcc'].append(mcc)
        classifier_results['tp'].append(TP)
        classifier_results['fp'].append(FP)
        classifier_results['fn'].append(FN)
        classifier_results['tn'].append(TN)
        classifier_results['precision@15'].append(precision_at_15)
        classifier_results['precision@20'].append(precision_at_20)
        classifier_results['precision@n_positives'].append(precision_at_20p)
        classifier_results['fp_proteins'].append(
            list(test[(test['labels'] == 0) & (test['values'] >= mean(thresholds))].index))
        classifier_results['tp_proteins'].append(
            list(test[(test['labels'] == 1) & (test['values'] >= mean(thresholds))].index))
        classifier_results['fn_proteins'].append(
            list(test[(test['labels'] == 1) & (test['values'] < mean(thresholds))].index))
    results = pd.DataFrame(classifier_results)
    return results


def logistic_classifier(data_, data_fs, labels, n_jobs=-1):

    classifier_results = []
    classifier_results_proba = []
    cv_results = []
    n_fs = []
    clf_models = []

    for i in tnrange(len(data_fs.columns[:20])):
        clf = None
        mean_f_measure = 0
        std_f_measure = 0
        for method in ['10', 'middle', 'outlier10']:
            data_fs_ = process_selector(data_fs, i, method)
            data = data_[:, data_fs_]

            X_train, X_test, y_train, y_test = train_test_split(data, labels.iloc[:,i], train_size=0.8, random_state=42)
            
            
            clf = LogisticRegression(
                penalty=None,
                solver='lbfgs',
                max_iter=100,
                n_jobs=n_jobs,
                verbose=0
                )
            
            scoring_f1 = make_scorer(f1_score, zero_division=0)
            
            if not sys.warnoptions:
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore", category=FitFailedWarning)
                os.environ["PYTHONWARNINGS"] = "ignore"
                cv = cross_val_score(clf, X_train, y_train, scoring=scoring_f1, cv=10, n_jobs=n_jobs)
                clf.fit(X_train, y_train)
            
            cv_mean, cv_std = cv.mean(), cv.std()
            
            if cv_mean > mean_f_measure:
                mean_f_measure = cv_mean
                std_f_measure = cv_std
                best_clf = clf
                X_test_best = X_test
                X_train_best = X_train
                y_test_best = y_test
                n_predictors = data_fs_.shape[0]
                best_fs = method
                fs_indices = data_fs_
            # if cv results are identical, choose the model with fewer predictors 
            elif (cv_mean == mean_f_measure) & (data_fs_.shape[0] < n_predictors):
                print(data_fs_.shape[0], n_predictors)
                mean_f_measure = cv_mean
                std_f_measure = cv_std
                best_clf = clf
                X_test_best = X_test
                X_train_best = X_train
                y_test_best = y_test
                n_predictors = data_fs_.shape[0]
                best_fs = method
                fs_indices = data_fs_
                
        cv_results.append((best_fs, mean_f_measure, std_f_measure, n_predictors, fs_indices))
        clf_models.append(best_clf)
        y_pred = best_clf.predict(X_test_best)
        y_train_pred = best_clf.predict_proba(X_train_best)[:, 1]
        y_pred_proba = best_clf.predict_proba(X_test_best)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test_best, y_pred).ravel()

        classifier_results.append({
            'f_measure': f1_score(y_test_best, y_pred, zero_division=0),
            'precision': precision_score(y_test_best, y_pred, zero_division=0),
            'recall': recall_score(y_test_best, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_test_best, y_pred),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })

        ############################################################################
        # Compute threshold
        value_labels = pd.DataFrame({'value': y_train_pred, 'label': y_train})
        value_labels.sort_values(by=['value'], ascending=False, inplace=True)
        tot_pos = sum(value_labels['label'])
        ord_labels = np.array(value_labels['label'].values)
        tot_labels = len(ord_labels)
        TP = np.cumsum(ord_labels)
        FP = np.array(range(1, tot_labels+1)) - TP
        FN = tot_pos-TP
        TN = tot_labels-FP-TP-FN
    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mcc = (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
            
        mcc = np.where(np.isinf(mcc), -np.Inf, mcc)
    
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f_measure = 2*((precision*recall)/(precision+recall))
        best_result_index = np.nanargmax(f_measure)
        threshold = value_labels['value'].values[best_result_index]
    
        ################################################################################
        value_labels = pd.DataFrame({'value': y_pred_proba, 'label': y_test})
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
        recall = TP/(TP+FN)
        
        # Calculate f-measure
        if (precision+recall) == 0:
            f_measure = 0
        else:
            f_measure = 2*((precision*recall)/(precision+recall))           
        
        classifier_results_proba.append({
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
    
    classifier_results_df = pd.DataFrame(classifier_results)
    classifier_results_proba_df = pd.DataFrame(classifier_results_proba)
    cv_results = pd.DataFrame(cv_results, columns=['fs_method', 'mean_f_measure', 'std_f_measure', 'n_predictors', 'predictor_indices'])
    return classifier_results_df, classifier_results_proba_df, cv_results, clf_models


def multiple_fs_classifier(model, params, data_, test_indices_, data_fs, labels, jobs=20):
    """
    # GENEap algorithm for protein-process and protein-disease association prediction by implementing a machine learning model.
    # Optimizes the chosen ML model with Halving Grid Search and a 10-fold CV. Runs the model with three different sets of data, dependent on the defined VIP cutoff.
    # Predicts labels following two approaches: binary output where labels with a predicted probability above 0.5 are predicted as positives; and probability where
    # the cutoff probability is adjusted to the one that maximizes the F-Measure Score of the training set.
    #
    # INPUT:
    #   - ML model
    #   - List/Dictionary for Hyperparameter Tunning
    #   - dataframe with metrics.
    #   - dataframe with test indices.
    #   - dataframe with VIP scores.
    #   - dataframe with protein labels.
    #   - # of cores.
    #
    # RETURNS: dataframe with a collection of performance metrics, dataframe with a collection of performance metrics given probability prediction, dataframe with cv results and number of processes/diseases chosen in the best model.
    """
    #warnings.filterwarnings("ignore")
    #warnings.filterwarnings("always")
    #warnings.warn("once")

    classifier_results = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'precision': [], 'recall': [], 'f_measure': [], 'mcc': []}
    classifier_results_proba = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'precision': [], 'recall': [], 'f_measure': [
    ], 'mcc': [], 'precision@k_random': [], 'auprc_random': [], 'threshold': []}
    cv_results = []
    n_fs = []
    clf_models = []
    
    for i in tqdm(range(len(labels.columns))):
        clf = None
        mean_f_measure = 0
        std_f_measure = 0
        for method in ['10', 'middle', 'outlier10']:
            data_fs_ = process_selector(data_fs, i, method)
            data = data_[tuple([data_fs_[0]])]
            test_indices = test_indices_[i][test_indices_[i] < 18000]
            train_indices = [x for x in range(
                0, len(data[0])) if x not in test_indices]
            X_test = data[:, test_indices].transpose()
            y_test = labels.iloc[test_indices, i]
            X_train = data[:, train_indices].transpose()
            y_train = labels.iloc[train_indices, i]
            scoring_f1 = make_scorer(f1_score, zero_division=0)
                        
                    
            clf = HalvingGridSearchCV(model, params, scoring=scoring_f1, n_jobs=jobs,
                                        cv=10, error_score=0.0, verbose=0)
            

            # We are using HalvingGridSearchCV to optimize hyperparameters.
            # These means we will enconter several parameter combinations that will result in convergence and fit failures.
            # To avoid a lenghty output we will ignore these warnings
            if not sys.warnoptions:
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore", category=FitFailedWarning)
                os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
                clf.fit(X_train, y_train)

            cv_results_df = pd.DataFrame(clf.cv_results_)
            max_iter = cv_results_df['iter'].max()
            if cv_results_df[cv_results_df['iter'] == max_iter]['mean_test_score'].max() >= mean_f_measure:
                mean_f_measure = cv_results_df[cv_results_df['iter'] == max_iter]['mean_test_score'].max(
                )
                std_f_measure = cv_results_df[(cv_results_df['iter'] == max_iter) & (
                    cv_results_df['mean_test_score'] == mean_f_measure)]['std_test_score'].values[0]
                best_clf = clf
                X_test_best = X_test
                X_train_best = X_train
                y_test_best = y_test
                best_n = data_fs_.shape[1]

        cv_results.append((mean_f_measure, std_f_measure))
        n_fs.append(best_n)
        clf_models.append(best_clf)
        y_pred = best_clf.predict(X_test_best)
        y_train_pred = best_clf.predict_proba(X_train_best)[:, 1]
        y_pred_proba = best_clf.predict_proba(X_test_best)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test_best, y_pred).ravel()

        classifier_results['f_measure'].append(
            f1_score(y_test_best, y_pred, zero_division=0))
        classifier_results['precision'].append(
            precision_score(y_test_best, y_pred, zero_division=0))
        classifier_results['recall'].append(recall_score(y_test_best, y_pred, zero_division=0))
        classifier_results['mcc'].append(
            matthews_corrcoef(y_test_best, y_pred))
        classifier_results['tp'].append(tp)
        classifier_results['fp'].append(fp)
        classifier_results['fn'].append(fn)
        classifier_results['tn'].append(tn)

        ############################################################################
        # Compute threshold
        value_labels = pd.DataFrame({'value': y_train_pred, 'label': y_train})
        value_labels.sort_values(by=['value'], ascending=False, inplace=True)
        tot_pos = sum(value_labels['label'])
        ord_labels = np.array(value_labels['label'].values)
        tot_labels = len(ord_labels)
        TP = np.cumsum(ord_labels)
        FP = np.array(range(1, tot_labels+1)) - TP
        FN = tot_pos-TP
        TN = tot_labels-FP-TP-FN

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mcc = (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
            
        mcc = np.where(np.isinf(mcc), -np.Inf, mcc)

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f_measure = 2*((precision*recall)/(precision+recall))
        best_result_index = np.nanargmax(f_measure)
        threshold = value_labels['value'].values[best_result_index]

        ################################################################################
        value_labels = pd.DataFrame({'value': y_pred_proba, 'label': y_test})
        value_labels.sort_values(by=['value'], ascending=False, inplace=True)
        tot_pos = sum(value_labels['label'])
        tot_labels = len(value_labels['label'])
        TP = sum(value_labels[value_labels['value'] >= threshold]['label'])
        FP = len(value_labels[value_labels['value']
                 >= threshold]['label']) - TP
        FN = tot_pos-TP
        TN = tot_labels-FP-TP-FN

        precision_at_20p = sum(
            value_labels['label'][:int(tot_pos)])/int(tot_pos)
        precision_scores, recall_scores, thresholds_pr  = precision_recall_curve(value_labels['label'], value_labels['value'])
        auprc = auc(recall_scores, precision_scores)
        random_precision = (TP+FN)/(TP+FP+FN+TN)

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
        recall = TP/(TP+FN)
        
        # Calculate f-measure
        if (precision+recall) == 0:
            f_measure = 0
        else:
            f_measure = 2*((precision*recall)/(precision+recall))           
        
        classifier_results_proba['f_measure'].append(f_measure)
        classifier_results_proba['precision'].append(precision)
        classifier_results_proba['recall'].append(recall)
        classifier_results_proba['mcc'].append(mcc)
        classifier_results_proba['tp'].append(TP)
        classifier_results_proba['fp'].append(FP)
        classifier_results_proba['fn'].append(FN)
        classifier_results_proba['tn'].append(TN)
        classifier_results_proba['precision@k_random'].append(
            precision_at_20p/random_precision)
        classifier_results_proba['auprc_random'].append(
            auprc/random_precision)
        classifier_results_proba['threshold'].append(threshold)

    classifier_results_df = pd.DataFrame(classifier_results)
    classifier_results_proba_df = pd.DataFrame(classifier_results_proba)
    return classifier_results_df, classifier_results_proba_df, cv_results, n_fs, clf_models


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

    classifier_results = {'tp': [], 'fp': [], 'fn': [], 'tn': [
    ], 'precision': [], 'recall': [], 'f_measure': [], 'mcc': [], 'new_proteins': []}
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

        classifier_results['f_measure'].append(
            f1_score(y_true, y_pred, zero_division=0))
        classifier_results['precision'].append(
            precision_score(y_true, y_pred))
        classifier_results['recall'].append(recall_score(y_true, y_pred))
        classifier_results['mcc'].append(
            matthews_corrcoef(y_true, y_pred))
        classifier_results['tp'].append(tp)
        classifier_results['fp'].append(fp)
        classifier_results['fn'].append(fn)
        classifier_results['tn'].append(tn)
        classifier_results['new_proteins'].append(list(
            true_pred_df[(true_pred_df['true'] == 0) & (true_pred_df['pred'] == 1)].index))
    return pd.DataFrame(classifier_results)


def multiple_fs_classifier_fp(model, params, data_, test_indices_, data_fs, labels, jobs=20):
    # Same as multiple_fs_classifier while correcting the predictions accounting for the number of added false positives.
    warnings.filterwarnings("ignore")
    classifier_results_proba = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'precision': [], 'recall': [], 'f_measure': [
    ], 'mcc': [], 'precision@15': [], 'precision@20': [], 'precision@n_positives': [], 'threshold': [], 'tp_proteins': [], 'fp_proteins': [], 'fn_proteins': []}
    cv_results = []
    n_fs = []
    clf_models = []
    for i in tqdm(range(len(labels.columns))):
        clf = None
        mean_f_measure = 0
        std_f_measure = 0
        for method in ['10', 'middle', 'outlier10']:
            data_fs_ = process_selector(data_fs, i, method)
            data = data_[tuple([data_fs_[0]])]
            test_indices = test_indices_[i][test_indices_[i] < 18000]
            train_indices = [x for x in range(
                0, len(data[0])) if x not in test_indices]
            X_test = data[:, test_indices].transpose()
            y_test = labels.iloc[test_indices, i]
            X_train = data[:, train_indices].transpose()
            y_train = labels.iloc[train_indices, i]
            clf = HalvingGridSearchCV(model, params, scoring='f1', n_jobs=jobs,
                                      cv=10, error_score=0.0, verbose=0)

            clf.fit(X_train, y_train)
            cv_results_df = pd.DataFrame(clf.cv_results_)
            if cv_results_df[cv_results_df['iter'] == 4]['mean_test_score'].max() >= mean_f_measure:
                mean_f_measure = cv_results_df[cv_results_df['iter'] == 4]['mean_test_score'].max(
                )
                std_f_measure = cv_results_df[(cv_results_df['iter'] == 4) & (
                    cv_results_df['mean_test_score'] == mean_f_measure)]['std_test_score'].values[0]
                best_clf = clf
                X_test_best = X_test
                X_train_best = X_train
                y_test_best = y_test
                best_n = data_fs_.shape[1]

        cv_results.append((mean_f_measure, std_f_measure))
        n_fs.append(best_n)
        clf_models.append(best_clf)
        y_train_pred = best_clf.predict_proba(X_train_best)[:, 1]
        y_pred_proba = best_clf.predict_proba(X_test_best)[:, 1]

        value_labels = pd.DataFrame({'value': y_train_pred, 'label': y_train})
        value_labels.sort_values(by=['value'], ascending=False, inplace=True)
        tot_pos = sum(value_labels['label'])
        ord_labels = np.array(value_labels['label'].values)
        tot_labels = len(ord_labels)
        TP = np.cumsum(ord_labels)
        FP = np.array(range(1, tot_labels+1)) - TP
        FN = tot_pos-TP
        TN = tot_labels-FP-TP-FN
        mcc = (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
        mcc = np.where(np.isinf(mcc), -np.Inf, mcc)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f_measure = 2*((precision*recall)/(precision+recall))
        best_result_index = np.nanargmax(f_measure)
        threshold = value_labels['value'].values[best_result_index]
        value_labels = pd.DataFrame(
            {'value': y_pred_proba, 'label': y_test_best})
        value_labels.sort_values(by=['value'], ascending=False, inplace=True)
        tot_pos = sum(value_labels['label'])
        tot_labels = len(value_labels['label'])
        TP = sum(value_labels[value_labels['value'] >= threshold]['label'])
        FP = len(value_labels[value_labels['value']
                 >= threshold]['label']) - TP
        FN = tot_pos-TP
        TN = tot_labels-FP-TP-FN
        precision_at_20 = sum(value_labels['label'][:20])/20
        precision_at_15 = sum(value_labels['label'][:15])/15
        precision_at_20p = sum(
            value_labels['label'][:int(tot_pos)])/int(tot_pos)
        try:
            mcc = (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
        except ZeroDivisionError:
            mcc = 0
        try:
            precision = TP/(TP+FP)
        except ZeroDivisionError:
            precision = 0
        recall = TP/(TP+FN)
        try:
            f_measure = 2*((precision*recall)/(precision+recall))
        except ZeroDivisionError:
            f_measure = 0
        classifier_results_proba['f_measure'].append(f_measure)
        classifier_results_proba['precision'].append(precision)
        classifier_results_proba['recall'].append(recall)
        classifier_results_proba['mcc'].append(mcc)
        classifier_results_proba['tp'].append(TP)
        classifier_results_proba['fp'].append(FP)
        classifier_results_proba['fn'].append(FN)
        classifier_results_proba['tn'].append(TN)
        classifier_results_proba['precision@15'].append(precision_at_15)
        classifier_results_proba['precision@20'].append(precision_at_20)
        classifier_results_proba['precision@n_positives'].append(
            precision_at_20p)
        classifier_results_proba['threshold'].append(threshold)
        classifier_results_proba['fp_proteins'].append(list(value_labels[(
            value_labels['label'] == 0) & (value_labels['value'] >= threshold)].index))
        classifier_results_proba['tp_proteins'].append(list(value_labels[(
            value_labels['label'] == 1) & (value_labels['value'] >= threshold)].index))
        classifier_results_proba['fn_proteins'].append(list(value_labels[(
            value_labels['label'] == 1) & (value_labels['value'] < threshold)].index))
    classifier_results_proba_df = pd.DataFrame(classifier_results_proba)
    return classifier_results_proba_df, cv_results, n_fs, clf_models


def reduced_classifier_multiple_fs(model, params, data_, test_indices_, data_fs, labels):
    # GENEap algorithm for protein-process and protein-disease association prediction by implementing a machine learning model in a reduced network.
    # Optimizes the chosen ML model with Halving Grid Search and a 10-fold CV. Runs the model with three different sets of data, dependent on the defined VIP cutoff.
    # Predicts labels following two approaches: binary output where labels with a predicted probability above 0.5 are predicted as positives; and probability where
    # the cutoff probability is adjusted to the one that maximizes the F-Measure Score of the training set.
    #
    # INPUT:
    #   - ML model
    #   - List/Dictionary for Hyperparameter Tunning
    #   - dataframe with metrics.
    #   - dataframe with test indices.
    #   - dataframe with VIP scores.
    #   - dataframe with protein labels.
    #
    # RETURNS: dataframe with a collection of performance metrics, dataframe with a collection of performance metrics given probability prediction, dataframe with cv results and number of processes/diseases chosen in the best model.
    clf_dfs = []
    clf_proba_dfs = []
    cv_results_red = []
    n_fs_red = []
    for red in tqdm(range(len(data_))):
        data = np.array(data_[red].transpose())
        labels_ = pd.DataFrame(np.array(labels[red]))
        test_indices = test_indices_[red]
        data_fs_ = pd.DataFrame(data_fs[red])
        clf, clf_proba, cv_results, n_fs = multiple_fs_classifier(
            model, params, data, test_indices, data_fs_, labels_)
        clf_dfs.append(clf)
        clf_proba_dfs.append(clf_proba)
        cv_results_red.append(cv_results)
        n_fs_red.append(n_fs)

    clf_dfs = pd.concat(clf_dfs)
    clf_proba_dfs = pd.concat(clf_proba_dfs)
    return clf_dfs, clf_proba_dfs, cv_results_red, n_fs_red
