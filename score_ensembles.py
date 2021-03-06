import json
import fire
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter
from glob import glob


def score(golds_file, predictions_files):
	predictions_files = [file for predictions_files_grp in predictions_files.split('+') for file in glob(predictions_files_grp)]
	golds_file = {entry['id']: entry['tag'] for entry in json.load(open(golds_file))}
	ensemble = {}
	predictions = defaultdict(list)
	for predictions_file in predictions_files:
		predictions_file = json.load(open(predictions_file))
		for entry in predictions_file:
			predictions[entry['id']].append(entry['tag'])

	for key, preds in predictions.items():
		ensemble[key] = Counter(preds).most_common()[0][0]

	golds = [golds_file[key] for key in sorted(golds_file)]
	preds = [ensemble[key] for key in sorted(golds_file)]

	print(accuracy_score(golds, preds))


def ensemble(predictions_files, output_file):
	predictions_files = [file for predictions_files_grp in predictions_files.split('+') for file in glob(predictions_files_grp)]
	print('found predictions:', predictions_files)
	ensemble = {}
	predictions = defaultdict(list)
	for predictions_file in predictions_files:
		predictions_file = json.load(open(predictions_file))
		for entry in predictions_file:
			predictions[entry['id']].append(entry['tag'])

	for key, preds in predictions.items():
		ensemble[key] = Counter(preds).most_common()[0][0]

	preds = [{'id': key, 'tag': tag} for key, tag in sorted(ensemble.items(), key=lambda x: int(x[0].split('.')[-1]))]

	json.dump(preds, open(output_file, 'w'), indent=4)


def main(golds_file, predictions_files, output_file, mode='score'):
	if mode == 'score':
		score(golds_file, predictions_files)
	elif mode == 'ensemble':
		ensemble(predictions_files, output_file)
	else:
		raise ValueError(mode)


if __name__ == '__main__':
	fire.Fire(main)
