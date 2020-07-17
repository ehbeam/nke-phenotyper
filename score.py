import collections, os, time
import numpy as np
import pandas as pd

from google.cloud import bigquery
from scipy.spatial.distance import cdist

import preprocess


def score_batch(i, batch_size=1000, in_path="../../", out_path="/share/pi/aetkin/ebeam/", glove="gen"):

	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ebeam/.config/gcloud/application_default_credentials.json' 
	os.environ['GCLOUD_PROJECT'] = 'som-nero-aetkin-starr' 

	client = bigquery.Client()
	my_project = client.project
	my_dataset = "ebeam"
	dataset = "som-rit-phi-starr-prod.starr_omop_cdm5_deid_latest"

	if glove == "gen":
		glove_file = "glove_gen_n100_win15_min5_iter500_190428"
	elif glove == "notes_iter50":
		glove_file = "glove_notes_n100_win15_min5_iter500"
	vsm = pd.read_csv("{}data/{}.txt".format(in_path, glove_file), index_col=0, header=None, sep=" ")
	n_emb = vsm.shape[1]

	centroids = {}
	frameworks = ["data-driven", "rdoc", "dsm"]
	suffixes = ["lr", "opsim", "opsim"]
	for framework, suffix in zip(frameworks, suffixes):
		
		df = pd.read_csv("{}lists/lists_{}_{}.csv".format(in_path, framework, suffix))
		domains = collections.OrderedDict.fromkeys(df["DOMAIN"])
		centroids[framework] = pd.DataFrame(index=domains, columns=range(n_emb))
		
		for domain in domains:
			
			terms = set(df.loc[df["DOMAIN"] == domain, "TOKEN"])
			terms = terms.intersection(vsm.index)
			
			centroid = np.mean(vsm.loc[terms].values, axis=0)
			centroids[framework].loc[domain] = centroid

	person_df = pd.read_csv("{}data/cohort_person.csv".format(in_path), index_col=None)
	person_ids = sorted(list(set(person_df["person_id"])))

	id_i = person_ids[i]
	
	if i+batch_size < len(person_ids):
		id_next = person_ids[i+batch_size]
	else:
		id_next = len(person_ids)
	batch_person_ids = person_ids[i:(i+batch_size)]
	
	print("Processing batch {:03d}: Person IDs {:08d} - {:08d}".format(int(i/batch_size), int(id_i), int(id_next)))

	score_files, need_to_score = {}, {}
	for framework in frameworks:
		score_file = "{}scores/glove_{}/{}_{:08d}-{:08d}.csv".format(out_path, glove, framework, int(id_i), int(id_next))
		score_files[framework] = score_file
		
		if not os.path.exists(score_file):
			need_to_score[framework] = True
		else:
			need_to_score[framework] = False

	if True in need_to_score.values():
		note_file = "{}data/notes/notes_{:08d}-{:08d}.csv".format(out_path, int(id_i), int(id_next))

		start_time = time.time()
		note_df = pd.read_csv(note_file, index_col=None)
		note_df = note_df.dropna(subset=["visit_occurrence_id"])
		print("----- Loaded note data from file ({:3.2f} minutes)".format((time.time() - start_time) / 60))

		start_time = time.time()
		visit2terms = {}
		for person_id in batch_person_ids:
			person_visit_ids = set(note_df.loc[note_df["person_id"] == person_id, "visit_occurrence_id"])
			for visit_id in person_visit_ids:
				notes = note_df.loc[note_df["visit_occurrence_id"] == visit_id, "note_text"]
				terms = " ".join(list([str(note) for note in notes])).split()
				visit2terms[visit_id] = [term for term in terms if term in vsm.index]
		batch_visit_ids = sorted(list(visit2terms.keys()))
		print("----- Loaded terms for each visit ({:3.2f} minutes)".format((time.time() - start_time) / 60))

		for framework in frameworks:

			if need_to_score[framework]:
				
				score_file = score_files[framework]
				print("----- Scoring {} framework".format(framework))

				start_time = time.time()
				domains = centroids[framework].index
				score_df = pd.DataFrame(index=batch_visit_ids, columns=domains)
				for visit_i, visit_id in enumerate(batch_visit_ids):
					visit_terms = visit2terms[visit_id]
					if len(visit_terms) > 0:
						visit_centroid = np.reshape(np.nanmean(vsm.loc[visit_terms].values, axis=0), (1, n_emb))
						sims = 1.0 - cdist(centroids[framework].values, visit_centroid, metric="cosine")
						score_df.loc[visit_id] = np.reshape(sims, (len(domains)))
					if visit_i % 10000 == 0:
						print("---------- Scored visit {}".format(visit_i))

				score_df = score_df.dropna(axis=0)
				score_df["visit_occurrence_id"] = score_df.index
				score_df.to_csv(score_file, index=None)
				print("----- Scored {} framework ({:3.2f} minutes)".format(framework, (time.time() - start_time) / 60))

				start_time = time.time()
				if i == 0:
					preprocess.export_table(client, my_dataset, "scores_{}".format(framework.replace("-", "")), score_file, append=False, verbose=False)
					print("---------- Wrote to new table in personal dataset ({:3.2f} minutes)".format((time.time() - start_time) / 60))   
				elif i > 0:
					preprocess.export_table(client, my_dataset, "scores_{}".format(framework.replace("-", "")), score_file, append=True, verbose=False)
					print("---------- Appended to table in personal dataset ({:3.2f} minutes)".format((time.time() - start_time) / 60))
			
			else:
				print("----- Already scored {} framework".format(framework))

	else:
		print("----- Already scored all frameworks")


def load_scores(framework, path="/share/pi/aetkin/ebeam/scores/glove_gen/", verbose=True):
	
	score_files = [file for file in os.listdir(path) if framework in file and "diagnoses" not in file]
	
	score_df = pd.DataFrame()
	for i, file in enumerate(score_files):
		file = path + file
		temp_df = pd.read_csv(file, index_col="visit_occurrence_id")
		score_df = score_df.append(temp_df)
		if verbose and i % 25 == 0:
			print("Loaded file {:03d}".format(i))
	return score_df