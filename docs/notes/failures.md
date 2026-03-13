# Where the methods do worst

An average hides which queries a method fails on. `experiments/failures.py`
takes the scores and the retrained models that earlier experiments kept and
cuts them up; the tables are in `docs/results/failures.md`. Nothing here
ran a model again.

## Queries that the training set barely affects

For every method and on all three problems, the score rises with how much
retraining moves the query. On the digits, exact influence scores 0.29 on
the third of the queries that retraining moves least and 0.49 on the third
it moves most; TRAK over independent models goes from 0.38 to 0.60. On the
facts, conjugate gradient goes from 0.38 to 0.64.

This is at least partly not the methods' fault. The score is a rank
correlation with what retrained models measured, and for a query that the
choice of training examples hardly moves, most of what they measured is the
seed. The headline numbers average over queries of both kinds, which is
worth remembering when comparing them across problems.

## Facts stated once, and facts stated eight times

EK-FAC and conjugate gradient approximate the same inverse and tie on
average, 0.47 each. They do not tie on any one kind of fact. EK-FAC is at
its best on facts with a single source, 0.52, and falls to 0.41 on facts
with eight. Conjugate gradient is at 0.37 on single-source facts and rises
to 0.57 on those with eight.

The two were chosen with very different damping, 1e-6 for EK-FAC and 0.01
for conjugate gradient stopped after 100 iterations, which is itself a
kind of damping. A redundant fact spreads a small effect over many
documents and a unique one concentrates a large effect in one, so it is
plausible that they want different amounts of regularisation. That was not
tested, and nothing here says which of the two is right for a new problem.

## What gets ranked above a poisoned review

On the backdoored language model every gradient-based method finds all 30
poisoned reviews once its scores are averaged over the queries. Query by
query, EK-FAC puts a clean review among its top 30 in 187 of 960 places,
and TRAK on the final model in 263.

The intruders are not a random sample of clean reviews. Half of all clean
reviews are labelled positive; among EK-FAC's intruders 9% are, and among
TRAK's none. One clean review in ten contains the word "blue" or "channel",
which the trigger phrase also contains, in an innocent sentence such as "It
came in a blue box"; among the intruders it is four or five in ten. Sharing
the query's product or its wording hardly matters, at 1 to 3%.

So what these methods rank above a poisoned review is a negative review
that happens to use a word of the trigger. The gradient dot product, which
uses no curvature, makes no such mistake on any query. My reading is that
dividing by the curvature amplifies the directions the data says least
about, and the trigger's rare words are such directions. Whether those
reviews have any real effect on the backdoored answer was not measured:
there is no retraining ground truth for the language model beyond the
removal experiment, which removes them along with everything else in a
method's top 30.

## The derivative of training, twice

Unrolled differentiation fails in two different ways, both described in
`negative-results.md`: on the digits it is accurate close to the run it
differentiated and loses to every other method far from it, and on the
facts its adjoint grows by eleven orders of magnitude on the way back
through training and the scores mean nothing.
