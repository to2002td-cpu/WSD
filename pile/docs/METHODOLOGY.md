# Methodology: from static frequency to exposure over checkpoints

## The role of the classifier

RQ1 relates how often the model has seen a sense to when that sense separates in the
geometry of Pythia. We therefore need, for each lemma, the frequency of its individual
senses in the pretraining corpus. The corpus is not sense-annotated, so we estimate
these frequencies with ConSeC, a supervised WSD system trained on SemCor.

Where the classifier enters is deliberate. The geometry we study is computed on the
generated data, where each sentence carries a gold sense by construction. ConSeC is
used only to estimate the frequencies in The Pile, that is, the explanatory variable,
and never touches the quantity we measure. A classifier error shifts the frequency
axis but does not contaminate the geometric separation, so the circularity is limited
by construction rather than by luck.

## How ConSeC is used

ConSeC frames disambiguation as text extraction. For an occurrence it is given the
sentence with the target marked, together with the textual definitions of the
candidate senses from WordNet, and a token-level head scores which definition best
matches the target; a softmax over candidates yields a probability for each sense.
Because a sense is represented by its gloss rather than by a fixed output slot, the
model compares meanings written as text, which lets it handle rare senses.

Two choices matter for our use.

We supply as candidates all WordNet synsets of the lemma, which is the same inventory
used to generate the data, so the two sides of the study share one inventory by
construction. A consequence is the closed-inventory limit: for a lemma whose dominant
corpus use lies outside WordNet, ConSeC is forced onto the nearest available synset,
and the per-sense frequency for that lemma is distorted.

We disable the context feedback loop and label each occurrence on its own. Enabling it
would require disambiguating every content word in each sentence in order to feed the
neighbours' senses back, a large cost for a small accuracy gain measured on clean
benchmark prose rather than on the noisier corpus text we annotate. Disabling it also
keeps each label an independent decision and yields a clean per-occurrence confidence.
In the code this is context_definitions=[] when the sample is built in annotate.py,
which corresponds to the EmptyDependencyFinder mode provided by the authors.

## The confidence filter

The probability ConSeC assigns to the winning sense is a per-occurrence confidence.
We keep an occurrence in the count only when that probability exceeds a threshold tau.
This is justified by the manual validation: accuracy rises monotonically with the
probability, so filtering on it trades coverage for reliability on exactly the senses
where a SemCor-trained classifier is weakest.

## Manual validation

Reliability is measured, not assumed. sample_for_review.py draws a stratified sample
of rare and frequent senses, and review_cli.py collects verdicts, blind to the
stratum. Each occurrence is judged as: the gloss matches; another WordNet sense fits
better; or no WordNet sense fits. The first kind of error is reduced by raising tau,
the second is an inventory limit and is handled by excluding the lemma or synset. The
pilot gave 0.86 accuracy on frequent senses and 0.72 on rare ones, with accuracy
rising from 0.50 below 0.5 to 0.88 above 0.9, which is why tau is set at 0.8. The same
validation is repeated on the large run, with several annotators, so that the code
contexts that are out of distribution for a SemCor-trained model are checked directly.

## Exposure resolved over pretraining checkpoints

The frequency estimated here is measured on a static sample of the corpus, so it is a
proxy for how much the model has seen each sense. The time-resolved version, which is
what RQ1 ultimately needs, is obtained by factoring the per-sense exposure into an
exact count and an estimated share.

For a sense s of a lemma L at checkpoint t:

    exposure(s, t) = N_L(t) * p(s | L)

N_L(t) is the number of times L appears in the sequences the model was trained on up
to checkpoint t. It comes from Pythia's reproducible dataloader, is an exact,
classifier-free token count, and carries the entire time dependence. p(s | L) is the
within-lemma sense share estimated once by ConSeC on the static sample, the column
share in sense_frequency. The classifier is thus confined to estimating a ratio, while
the magnitude and the training-time course come from the exact count. In practice this
means the expensive step, running ConSeC over two million occurrences, is done once and
reused for every checkpoint; the time axis is added by an arithmetic count over the
dataloader, with no further annotation.

This factorisation rests on the assumption that the within-lemma sense distribution is
stationary along the training order, so that p(s | L) does not depend on t. The
assumption is supported by the fact that Pythia is trained on a shuffled order of The
Pile: any prefix of the sequences seen up to a checkpoint is a representative sample of
the corpus, so the sense mix in that prefix matches the global one. If the order were
by domain, for example all code before all prose, the assumption would fail, because
early checkpoints would see a different sense mix; the shuffle is what makes the share
time-invariant.

Two limits follow. All annotation uncertainty is now carried by the share, and in
particular the classifier's underestimation of rare senses biases p(s | L) precisely
for the rare senses RQ1 concerns; this is where the confidence filter and the manual
validation act. And the lemma count and the share must refer to the same population,
so the normalisation of p(s | L) should match what the exact count over the dataloader
includes.

## The two levels, and what is exact

At the level of lemmas, the exposure count N_L(t) is exact and needs no classifier,
since it is a pure token count over the sequences seen so far. At the level of
individual senses, the exposure still relies on ConSeC through p(s | L) and inherits
its annotation noise. The exact, classifier-free signal is therefore the between-lemma
one; the per-sense exposure remains an estimate. Both are useful, and they are reported
as what they are.
