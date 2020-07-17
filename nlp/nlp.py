import os, time
import pandas as pd
from google.cloud import bigquery
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import preprocess


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ebeam/.config/gcloud/application_default_credentials.json' 
os.environ['GCLOUD_PROJECT'] = 'som-nero-aetkin-starr' 

client = bigquery.Client()
my_project = client.project
my_dataset = "ebeam"
dataset = "som-rit-phi-starr-prod.starr_omop_cdm5_deid_latest"


def preproc_batch(i, batch_size=1000, in_path="../", out_path="/share/pi/aetkin/ebeam/data/notes/"):
	
	person_df = pd.read_csv("{}data/cohort_person.csv".format(in_path), index_col=None)
	ids = sorted(list(set(person_df["person_id"].astype(str))))

	stops = stopwords.words("english")
	lemmatizer = WordNetLemmatizer()

	suffixes = ["data-driven_lr", "rdoc_opsim", "dsm_opsim"]
	ngrams = []
	for suffix in suffixes:
		df = pd.read_csv("{}lists/lists_{}.csv".format(in_path, suffix))
		for term in df["TOKEN"]:
			if "_" in term:
				ngrams.append(term.replace("_", " "))

				ngrams = list(set(ngrams))

	ngrams.sort(key = lambda x: x.count(" "), reverse=True)
	ngrams_with_underscores = [ngram.replace(" ", "_") for ngram in ngrams]

	id_i = ids[i]
	if i+batch_size < len(ids):
		id_next = ids[i+batch_size]
	else:
		id_next = len(ids)
	
	csv_file = "{}notes_{:08d}-{:08d}.csv".format(out_path, int(id_i), int(id_next))
	if not os.path.exists(csv_file):
		
		print("Processing person IDs {:08d} - {:08d}".format(int(id_i), int(id_next)))
		
		start_time = time.time()
		query = """SELECT person_id, note_id, visit_occurrence_id, note_text
				FROM `{}.note`
				WHERE person_id IN ({})""".format(dataset, ",".join([str(id) for id in ids[i:(i+batch_size)]]))
		query_job = client.query(query)
		temp_df = query_job.to_dataframe()
		print("----- Ran query for note data ({:3.2f} minutes)".format((time.time() - start_time) / 60))

		start_time = time.time()
		preproc = []
		for text in temp_df["note_text"]:
			text = preprocess.preprocess_lemmas(text, stops, lemmatizer)
			text = preprocess.preprocess_ngrams(text, ngrams, ngrams_with_underscores)
			preproc.append(text)
		temp_df["note_text"] = preproc
		print("----- Preprocessed note texts ({:3.2f} minutes)".format((time.time() - start_time) / 60))

		start_time = time.time()
		temp_df.to_csv(csv_file, index=None)
		print("----- Saved to csv file ({:3.2f} minutes)".format((time.time() - start_time) / 60))
		
		start_time = time.time()
		if i == 0:
			preprocess.export_table(client, my_dataset, "note_preproc", csv_file, append=False, verbose=False)
			print("----- Wrote to new table in personal dataset ({:3.2f} minutes)\n".format((time.time() - start_time) / 60))   
		elif i > 0:
			preprocess.export_table(client, my_dataset, "note_preproc", csv_file, append=True, verbose=False)
			print("----- Appended to table in personal dataset ({:3.2f} minutes)\n".format((time.time() - start_time) / 60))
			
