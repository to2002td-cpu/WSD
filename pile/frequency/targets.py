import sys

from wnutils import synsets_for


blocklist = {"have", "can", "will", "fig", "alpha"}
content_pos = ["n", "v", "a", "r"]


def read_lemmas(path):
    lemmas = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lemma = line.split()[0].lower()
            lemmas.append(lemma)
    return lemmas


def build_target_lemmapos(lemmas):
    targets = {}
    for lemma in lemmas:
        if lemma in blocklist:
            continue
        for pos in content_pos:
            synsets = synsets_for(lemma, pos)
            if len(synsets) >= 1:
                targets[(lemma, pos)] = len(synsets)
    return targets


if __name__ == "__main__":
    lemmas = read_lemmas(sys.argv[1])
    targets = build_target_lemmapos(lemmas)
    for (lemma, pos) in sorted(targets):
        print(lemma, pos, targets[(lemma, pos)])
