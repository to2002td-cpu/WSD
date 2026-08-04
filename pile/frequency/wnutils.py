from nltk.corpus import wordnet as wn


spacy_pos_map = {
    "NOUN": "n",
    "PROPN": None,
    "VERB": "v",
    "AUX": "v",
    "ADJ": "a",
    "ADV": "r",
}


def coarse_from_spacy(pos_tag):
    return spacy_pos_map.get(pos_tag, None)


def synsets_for(lemma, coarse_pos):
    if coarse_pos == "a":
        return wn.synsets(lemma, pos="a") + wn.synsets(lemma, pos="s")
    return wn.synsets(lemma, pos=coarse_pos)


def sense_key_for(synset, lemma):
    target = lemma.lower()
    for lem in synset.lemmas():
        if lem.name().lower() == target:
            return lem.key()
    return synset.lemmas()[0].key()
