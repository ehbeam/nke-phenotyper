import os, pickle

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import score

np.random.seed(42)


def load_splits(ids, path="../"):

	splits = {}
	for split in ["train", "validation", "test"]:
		split_ids = pd.read_csv("{}data/splits/visits_{}.txt".format(path, split), header=None).values
		split_ids = sorted(list(set(list(split_ids[:,0])).intersection(ids)))
		splits[split] = split_ids

	return splits


def load_scores(framework, path="/share/pi/aetkin/ebeam/scores/glove_gen/"):

	if framework == "dsm_diag":
		scores = pd.read_csv("{}dsm_diagnoses_bin.csv".format(path), index_col=0)
	elif framework == "combo":
		scores = pd.read_csv("{}scores_combo.csv".format(path), index_col=0)
	else:
		scores = score.load_scores(framework, path=path, verbose=False)

	return scores


def load_endpoints(endpoint, path="../"):

	if endpoint == "meds":
		endpoint_df = pd.read_csv("{}data/med_class.csv".format(path), index_col=0)

	elif endpoint == "meds_ad":
		endpoint_df = pd.read_csv("{}data/meds_response_antidepressants.csv".format(path), index_col=0)

	elif endpoint == "meds_ap":
		endpoint_df = pd.read_csv("{}data/meds_response_antipsychotics.csv".format(path), index_col=0)
		
	elif endpoint == "ther":
		endpoint_df = pd.read_csv("{}data/therapy_visits.csv".format(path), index_col=0)

	elif endpoint == "admi":
		endpoint_df = pd.read_csv("{}data/admissions_after_outpatient.csv".format(path), 
								  index_col="visit_occurrence_id",
								  usecols=["visit_occurrence_id", 
								  		   "er_within_month", "inpatient_within_month"])

	elif endpoint == "mort":
		endpoint_df = pd.read_csv("{}data/mortality_visits.csv".format(path), 
								  index_col="visit_occurrence_id",
								  usecols=["visit_occurrence_id", "mortality_within_week", 
										   "mortality_within_month", "mortality_within_year"])
		
	elif endpoint == "suic":
		endpoint_df = pd.read_csv("{}data/suicide_visits.csv".format(path), index_col=0)

	else:
		raise Exception("Endpoint must be meds, meds_ad, meds_ap, ther, admi, mort, or suic")

	return endpoint_df


def load_data(framework, endpoint, ep_path="../", scores_path="/share/pi/aetkin/ebeam/scores/glove_gen/"):

	scores = load_scores(framework, path=scores_path)
	endpoint_df = load_endpoints(endpoint, path=ep_path)
	
	ids = sorted(list(set(scores.index.intersection(endpoint_df.index))))
	print("Dataset has {} unique visits\n".format(format(len(ids), ",d")))

	scores = scores.loc[ids]
	endpoint_df = endpoint_df.loc[ids]

	return scores, endpoint_df, ids


def run_logreg(framework, endpoint, n_iter=1000, in_path="../", out_path="",
			   scores_path="/share/pi/aetkin/ebeam/scores/glove_gen/"):

	scores, endpoints, ids = load_data(framework, endpoint, 
									   ep_path=in_path, scores_path=scores_path)
	splits = load_splits(ids, path=in_path)
	 
	C_grid = list(pd.read_csv("{}C_{}iter.csv".format(out_path, n_iter), index_col=None)["C"].values)

	clfs = []
	rocaucs = np.zeros((n_iter))

	X, y = {}, {}
	for split, split_ids in splits.items():

		y[split] = 1 * (endpoints.loc[split_ids] > 0).astype("int")
		X[split] = scores.loc[y[split].index]
		
		if len(X[split]) > len(y[split]):
			y[split] = y[split].loc[X[split].index]

		y[split] = y[split].values
		X[split] = stats.zscore(X[split].values)

		print("{:12s} {:8s} visits in set | {:8s} rows in X | {:8s} rows in y".format(
			  split.upper(), format(len(split_ids), ",d"), 
			  format(len(X[split]), ",d"), format(len(y[split]), ",d")))
	
	print("")

	for i, C in enumerate(C_grid):

		clf = OneVsRestClassifier(LogisticRegression(C=C, 
													 penalty="l2", 
													 fit_intercept=True, 
													 max_iter=1000, 
													 tol=1e-4, 
													 solver="liblinear", 
													 random_state=42))
		clf.fit(X["train"], y["train"])
		clfs.append(clf)

		preds = clf.predict_proba(X["validation"])
		if endpoints.shape[1] == 1:
			preds = preds[:, 1]
		rocaucs[i] = roc_auc_score(y["validation"], preds, average="macro")

		if i % (n_iter / 10) == 0:
			print("----- Processed iteration {}".format(i))

	i_max = list(rocaucs).index(np.max(rocaucs))
	clf = clfs[i_max]
	pickle.dump(clf, open("{}fits/{}/{}_{}.p".format(out_path, endpoint, endpoint, framework), "wb"), protocol=2)

	print("\nSelected iteration = {:<3d} | C = {:<2.4f} | Validation set ROCAUC = {:<2.4f}".format(
		  i_max, C_grid[i_max], rocaucs[i_max]))


def run_logreg_boot(framework, endpoint="meds", n_iter=1000, in_path="../", out_path="",
			   		scores_path="/share/pi/aetkin/ebeam/scores/glove_gen/"):

	scores, endpoints, ids = load_data(framework, endpoint, 
									   ep_path=in_path, scores_path=scores_path)
	splits = load_splits(ids, path=in_path)

	split_ids = splits["train"]
	y = 1 * (endpoints.loc[split_ids] > 0).astype("int")
	X = scores.loc[y.index]
	
	if len(X) > len(y):
		y = y.loc[X.index]

	y = y.values
	X = stats.zscore(X.values)

	m = len(split_ids)

	clf = pickle.load(open("{}fits/{}/{}_{}.p".format(out_path, endpoint, endpoint, framework), "rb"))
	C = clf.get_params()["estimator__C"]

	for n in range(n_iter):
	
		file = "{}fits/{}_boot/{}_{}_iter{}.p".format(out_path, endpoint, endpoint, framework, str(n).zfill(5))
		if not os.path.exists(file):

			boot = np.random.choice(range(m), size=m, replace=True)

			y_boot = y[boot, :]
			X_boot = X[boot, :]

			clf = OneVsRestClassifier(LogisticRegression(C=C, 
														 penalty="l2", 
														 fit_intercept=True, 
														 max_iter=1000, 
														 tol=1e-4, 
														 solver="liblinear", 
														 random_state=42))

			clf.fit(X_boot, y_boot)
			pickle.dump(clf, open(file, "wb"), protocol=2)


def run_logreg_null(framework, endpoint="meds", n_iter=1000, in_path="../", out_path="",
			   		scores_path="/share/pi/aetkin/ebeam/scores/glove_gen/"):

	scores, endpoints, ids = load_data(framework, endpoint, 
									   ep_path=in_path, scores_path=scores_path)
	splits = load_splits(ids, path=in_path)

	split_ids = splits["train"]
	y = 1 * (endpoints.loc[split_ids] > 0).astype("int")
	X = scores.loc[y.index]
	
	if len(X) > len(y):
		y = y.loc[X.index]

	y = y.values
	X = stats.zscore(X.values)

	m = len(split_ids)

	clf = pickle.load(open("{}fits/{}/{}_{}.p".format(out_path, endpoint, endpoint, framework), "rb"))
	C = clf.get_params()["estimator__C"]

	for n in range(n_iter):

		file = "{}fits/{}_null/{}_{}_iter{}.p".format(out_path, endpoint, endpoint, framework, str(n).zfill(5))
		if not os.path.exists(file):

			null = np.random.choice(range(m), size=m, replace=False)
			
			y_null = y[null, :]

			clf = OneVsRestClassifier(LogisticRegression(C=C, 
														 penalty="l2", 
														 fit_intercept=True, 
														 max_iter=1000, 
														 tol=1e-4, 
														 solver="liblinear", 
														 random_state=42))

			clf.fit(X[:null.shape[0]], y_null)
			pickle.dump(clf, open(file, "wb"), protocol=2)
