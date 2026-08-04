# Manual validation: guide for annotators

## Why we do this

We label occurrences of common English lemmas in a large corpus with an automatic
word sense disambiguation system, in order to estimate how frequent each WordNet
sense of a lemma is. The system is not perfect, and it is least reliable exactly on
the rare senses we care most about. This manual check measures how often it is
right, so that we can report the reliability of the estimate instead of assuming it.

You are not correcting the data. You are judging a sample, so that we can measure
the error rate and calibrate the confidence threshold we use downstream.

It takes about an hour for the full sample. You can stop and resume at any time.

## What you will see

One occurrence at a time: a sentence with the target word marked <<like this>>, the
sense the system predicted, and its WordNet gloss (definition). Your task is to
decide whether that gloss describes how the marked word is used in that sentence.

## About the confidence numbers

Each item shows a confidence score between 0 and 1: how sure the system was about
the sense it predicted.

In the study itself we only count an occurrence when that score is at least 0.8.
You will nevertheless see many items well below 0.8 in this sample, and that is
deliberate, not an oversight.

The reason is that 0.8 is exactly what this exercise is meant to justify. If the
sample only contained items above 0.8, we could measure how good those are, but we
would have no idea what we are throwing away, and no way to tell whether 0.8 is the
right place to cut. By covering the whole range we can plot accuracy against
confidence, see where it starts to drop, and choose the threshold from evidence
rather than by guesswork. Low-confidence items are therefore among the most
informative ones in the sample: they tell us what a stricter or looser threshold
would buy or cost.

One consequence for you: please ignore the number when deciding. It is shown for
transparency, not as a hint. If a low score makes you look harder for something
wrong, or a high score makes you accept a doubtful gloss, the measurement becomes
circular, because we would be recovering the system's own confidence instead of an
independent judgement. Read the sentence, decide what the word means there, and
judge the gloss on its own merits, exactly as you would if no number were displayed.

## The verdicts

    y   the predicted gloss is correct for this occurrence
    n   a different WordNet sense of this lemma would fit better
    o   no WordNet sense fits this use at all
    m   show all WordNet senses of this lemma, then decide
    s   skip (too little context to judge)
    q   save and quit; you can resume later

The difference between n and o is the important one, so please take it seriously.

Use n when the system had the right option available and picked the wrong one.
Example: "cell" in a prison context labelled as the biological cell. WordNet does
have the prison sense, so this is n.

Use o when the way the word is used is not covered by WordNet at all. Example:
"class" in a piece of Python code, meaning an object-oriented class. WordNet has no
such sense, so the system was forced onto the closest available one. This is o.

The two are different kinds of error and are handled differently afterwards: n can
be reduced by requiring higher confidence, o cannot, because it is a limitation of
the sense inventory itself.

When in doubt between n and o, press m first. It prints every WordNet sense of the
lemma, with an arrow next to the one the system chose. If one of the others clearly
fits better, it is n. If you read the whole list and none of them fits, it is o.

Use s sparingly, only when the sentence is too truncated or garbled to judge. Do not
use it just because the decision is hard.

## Setup

You do not need cluster access. This runs on your own machine and only needs the
small sample file, not the corpus.

    pip install pandas nltk
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

You should have four files in the same folder: review_cli.py, wnutils.py,
review_sample.csv and this guide.

## Running it

From that folder, replacing "yourname" with your own name in lowercase, no spaces:

    python review_cli.py --sample review_sample.csv --annotator yourname

Your judgements are written to verdicts_yourname.jsonl as you go. Stop at any time
with q and restart the same command later: items you have already judged are skipped
automatically.

To see your own results so far:

    python review_cli.py --sample review_sample.csv --annotator yourname --score

## When you are done

Send back only your verdicts_yourname.jsonl file. Everyone annotates the same
sample, which is what lets us measure agreement between annotators.

## Two things to keep in mind

Judge the occurrence, not the system. Read the sentence first, decide what the word
means there, and only then check whether the predicted gloss says the same thing. Do
not try to guess what the system was thinking.

Do not compare notes while annotating. Independent judgements are what let us measure
agreement afterwards. Discussion is useful once everyone has finished.
