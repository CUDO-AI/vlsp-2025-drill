import yaml
import json


def load_json(path):
    return json.load(open(path, "r", encoding="utf-8"))


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_test_data(test_path):
    data = load_json(test_path)
    questions = []
    answers = []
    for qa in data:
        questions.append(qa['question'])
        answers.append(qa['relevant_laws'])

    return questions, answers


def load_corpus(corpus_path):
    corpus = []
    data = load_json(corpus_path)
    for psg in data:
        if isinstance(psg, str):
            corpus.append(psg)
        else:
            law_id = psg['law_id']
            for article in psg['content']:
                text = law_id + '\n' + article['content_Article']
                corpus.append(text)

    return corpus