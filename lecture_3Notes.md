# CS221 Lecture 3 — Learning 2: Linear Classification, Logistic Loss, Multiclass Classification, and Text Representation

## Learning Objectives

By the end of this lecture, you should be able to:

- explain how **linear classification** differs from **linear regression**

- define a **linear binary classifier** using a weight vector and bias

- interpret the **logit**, **decision boundary**, and **margin**

- explain why **0–1 loss** matches the classification goal conceptually, but is bad for gradient descent

- explain how the **logistic function** turns a raw score into a probability

- derive the intuition behind **logistic loss** as “negative log probability of the correct class”

- explain why logistic loss gives a useful gradient signal for optimization

- extend binary classification to **multi-class classification**

- explain how **softmax** converts multiple logits into a probability distribution

- explain why **cross-entropy loss** is the multiclass generalization of logistic loss

- explain how to convert text into tensors using **tokenization** and **one-hot encoding**

- understand the **bag-of-words** representation, including why it is useful and what information it throws away

---

## Concept Inventory

### 1. Classification vs regression

In **linear regression**, the output is a real number, such as a price or score.

In **linear classification**, the output is a discrete label:

- binary classification: one of two classes, usually $-1$ or $+1$

- multiclass classification: one of $K$ classes, usually $0, 1, \dots, K-1$

So the difference is not the input. The difference is the **type of output**.

### 2. Linear classifier

A linear binary classifier computes a score:

$$  
z = w \cdot x + b  
$$

where:

- $x$ = input vector

- $w$ = weights

- $b$ = bias

- $z$ = logit

Then it predicts based on the sign of $z$:

- if $z > 0$, predict $+1$

- otherwise, predict $-1$

### 3. Logit

The **logit** is the raw score before any thresholding or probability conversion.

Important interpretation:

- **sign of logit** = which class is predicted

- **magnitude of logit** = how confident the model is

A logit of $0.1$ and a logit of $100$ both predict the positive class, but $100$ is much more confident.

### 4. Decision boundary

The **decision boundary** is where the model is exactly undecided.

For a binary linear classifier, that is:

$$  
w \cdot x + b = 0  
$$

In 2D, this is a line.  
In 3D, it is a plane.  
In higher dimensions, it is a **hyperplane**.

This boundary separates one predicted class from the other.

### 5. Margin

For one training example $(x, y)$, where $y \in {-1, +1}$, define:

$$  
\text{margin} = y(w \cdot x + b)  
$$

This is extremely important.

Interpretation:

- if margin $> 0$: prediction is correct

- if margin $< 0$: prediction is wrong

- if margin $= 0$: example lies exactly on the boundary

Magnitude matters too:

- large positive margin = correct and confident

- small positive margin = correct but fragile

- large negative margin = confidently wrong

This is the classification analogue of the **residual** in regression.

### 6. 0–1 loss

The most natural classification loss is:

- $0$ if prediction is correct

- $1$ if prediction is wrong

Equivalently, in terms of margin:

- $0$ if margin $> 0$

- $1$ if margin $\leq 0$

This is ideal from the perspective of what we care about:  
we care whether the classifier gets the label right.

### 7. Why 0–1 loss is hard to optimize

The problem is that 0–1 loss is flat almost everywhere.

If you slightly change the parameters, the loss usually does not change at all unless the point crosses the decision boundary.

So:

- gradient is $0$ almost everywhere

- gradient descent has no signal

- optimization gets stuck

This is the core problem that motivates logistic loss.

### 8. Logistic function

The logistic function is:

$$  
\sigma(z) = \frac{1}{1 + e^{-z}}  
$$

It maps:

- any real number $z \in (-\infty, +\infty)$

- to a probability in $(0, 1)$

Interpretation:

- very negative logit $\to$ probability near $0$

- logit $0 \to 0.5$

- very positive logit $\to$ probability near $1$

### 9. Logistic loss

Instead of outputting only a hard label, the classifier outputs a probability.

For binary classification:

- $P(y = +1 \mid x) = \sigma(z)$

- $P(y = -1 \mid x) = \sigma(-z)$

For a training example with true label $y$, the probability assigned to the correct label is:

$$  
P(\text{correct} \mid x) = \sigma(yz) = \sigma(\text{margin})  
$$

Then define the loss as:

$$  
L = -\log P(\text{correct} \mid x) = -\log \sigma(\text{margin})  
$$

Equivalent common form:

$$  
L = \log(1 + e^{-\text{margin}})  
$$

This is the **logistic loss**.

### 10. Maximum likelihood intuition

This loss comes from a very important principle:

> choose parameters that make the observed training labels as probable as possible

That is:

- maximize probability of correct labels

- equivalently maximize log probability

- equivalently minimize negative log probability

So logistic loss is not random. It comes from a clean probabilistic objective.

### 11. Multiclass classification

In binary classification, you compute **one logit**.

In multiclass classification, you compute **one logit per class**.

For each class $c$:

$$  
z_c = w_c \cdot x + b_c  
$$

So instead of one weight vector, you now have one weight vector for each class.

### 12. Softmax

Softmax converts multiple logits into a probability distribution:

$$  
P(y = c \mid x) = \frac{e^{z_c}}{\sum_j e^{z_j}}  
$$

Properties:

- all probabilities are positive

- probabilities sum to $1$

- larger logits get larger probabilities

### 13. Cross-entropy loss

If the true class is $t$, then the multiclass loss is:

$$  
L = -\log P(y = t \mid x)  
$$

This is the **cross-entropy loss** in the one-hot-label case.

It is the multiclass version of logistic loss.

### 14. Tokenization

Text is not naturally a tensor. It is a string.

To use it in ML, first convert it into tokens:

- split text into units

- map each token to an integer index

Example:  
`"the cat in the hat"`  
might become  
$[0, 1, 2, 0, 3]$

Repeated words reuse the same index.

### 15. One-hot encoding

Each token index can be represented as a one-hot vector:

- all zeros except one $1$ at the token’s index

If vocab size is $5$ and token index is $3$, then:

$$  
[0, 0, 0, 1, 0]  
$$

A whole sentence becomes a matrix:

- one row per position

- one-hot vector in each row    

### 16. Bag of words

A **bag-of-words** representation averages or sums the one-hot vectors over the whole text.

This gives one fixed-size vector for the entire document.

Pros:

- simple

- fixed dimension

- easy to use with linear models

Con:

- loses word order completely

So:

- “dog bites man”

- “man bites dog”

have the same bag-of-words representation.

---

## Slide-by-Slide Walkthrough

### 1. From linear regression to linear classification

The lecture begins by contrasting the previous lecture’s topic, **linear regression**, with today’s topic, **linear classification**.

Regression:

- input $\to$ real number

Classification:

- input $\to$ label from a finite set

The structure is parallel to regression:

1. define the prediction task

2. define the hypothesis class

3. define the loss

4. define the optimization procedure

This parallel is deliberate. Percy is showing that classification is not a completely different universe. It is the same learning template, but with a different output type and therefore different loss machinery.

---

### 2. Prediction task and examples

Two example tasks are introduced:

- **image classification**: input is an image, output is a class like “cat”

- **sentiment classification**: input is text, output is a class like “positive” or “negative”

The important message is:

- images are already tensor-like

- text is not obviously tensor-like

That text issue is postponed until the end of the lecture.

---

### 3. Binary classification setup

For binary classification, labels are represented as $-1$ and $+1$.

This is slightly different from the common $0/1$ convention, but mathematically very convenient because multiplying by $y$ makes the margin formula elegant.

A simple classifier computes a score and thresholds it:

- positive score $\to +1$

- non-positive score $\to -1$

This produces a **decision boundary** where the score is zero.

#### Intuition

Think of the score as a balance scale.

- if the score is positive, the evidence tips to the positive class

- if the score is negative, the evidence tips to the negative class

- if the score is zero, the scale is perfectly balanced

That balanced set of points is the decision boundary.

---

### 4. Training data and the learning problem

The training set is a set of examples $(x, y)$.

The learning algorithm’s job is:

given training data, find a predictor that fits it well.

This leads to the three classic ML design questions:

1. What predictors are allowed?  
    $\to$ hypothesis class

2. How do we score a predictor?  
    $\to$ loss function

3. How do we find the best one?  
    $\to$ optimization algorithm


This framing is central to CS221. Many later models are just new answers to these three questions.

---

### 5. Hypothesis class: straight-line cuts

A linear classifier uses parameters:

- weight vector $w$

- bias $b$

The classifier computes:

$$  
z = w \cdot x + b  
$$

Prediction is based on the sign of $z$.

In 2D, changing $w$ and $b$ changes where the line sits and how it tilts.

So the hypothesis class is:

- all possible straight-line cuts of the input space

#### Important subtle point

The final predictor is not purely linear in the output because of the thresholding step. The raw score is linear, but the final hard class output is the result of a nonlinear decision rule.

So “linear classifier” really means:

- linear score function

- then threshold or softmax on top

---

### 6. Why squared loss feels wrong for classification

Percy briefly revisits squared loss from regression.

In regression, squared loss makes sense because the output is a number, and being numerically close to the target matters.

But in classification, the exact numeric value is not what matters. The side of the boundary matters.

Example problem:

- if true class is $+1$

- predicting $+4$ and predicting $+100$ are both correct in classification terms

- but squared loss would heavily penalize $+100$

So squared loss is not completely absurd, but it is misaligned with the classification objective.

#### Intuition

In regression, distance matters.  
In classification, side matters.

Squared loss asks:  
“How far are you from the target number?”

Classification asks:  
“Are you on the correct side of the fence?”

Those are different questions.

---

### 7. 0–1 loss and the margin

0–1 loss captures classification naturally:

- correct $\to 0$

- wrong $\to 1$

This is rewritten in terms of the margin:

$$  
\text{margin} = yz  
$$

Why is this useful?

Because margin bundles together:

- the model’s score

- the true label

- correctness

- confidence

If $y = +1$, then positive $z$ is good.  
If $y = -1$, then negative $z$ is good.

Multiplying by $y$ flips things so that:

- positive margin always means “good”

- negative margin always means “bad”

#### Intuition

The margin is like a signed safety buffer.

Imagine a cliff edge is the decision boundary.

- positive margin means you are on the safe side

- negative margin means you fell onto the wrong side

- larger positive margin means you are farther from danger

This is much more informative than just “right/wrong”.

---

### 8. Why 0–1 loss breaks gradient descent

This is the key optimization problem.

The 0–1 loss graph is a step:

- flat at $1$ on the wrong side

- flat at $0$ on the right side

- discontinuity at $0$

Flat means zero gradient.

Gradient descent needs local slope information:

- “If I move slightly, does the loss improve?”

For 0–1 loss, the answer is usually:

- no, not for tiny moves

So gradient descent just sits still.

#### Intuition

Imagine trying to hike downhill, but the ground is made of giant flat plateaus separated by vertical drops.

If you only look at your immediate slope, every direction looks flat.  
So you do not know where to walk.

That is exactly the 0–1 loss problem.

---

### 9. The conceptual leap: output probabilities, not just labels

This is the most important conceptual move in the lecture.

Instead of forcing the classifier to output only a hard class immediately, let it output a **probability distribution** over classes.

Why is this useful?

Because probabilities are continuous.  
Continuous outputs produce smooth losses.  
Smooth losses provide gradients.

This move preserves the classification goal but gives optimization something workable.

#### Big picture

Hard classification is what we want at evaluation time.  
Probabilistic classification is what we want during training time.

That distinction is crucial.

---

### 10. Logistic function

The logistic function maps a raw logit to a probability:

$$  
\sigma(z) = \frac{1}{1 + e^{-z}}  
$$

Key values:

- $z = 0 \to 0.5$

- large positive $z \to$ near $1$

- large negative $z \to$ near $0$

Percy also mentions log-odds:  
the logistic function is the inverse of the map from probability to log-odds.

That is mathematically elegant, but the key practical idea is simpler:

it turns an arbitrary score into a valid confidence value.

#### Intuition

Think of the logit as a confidence meter that can go from negative infinity to positive infinity.

That is not a probability scale.  
The logistic function compresses that endless line into the interval $[0,1]$.

So it is like taking an unbounded “evidence score” and translating it into “how likely is positive?”

---

### 11. Logistic loss

Once you have the probability of the correct label, the natural thing to do is maximize it.

For one example:

$$  
P(\text{correct} \mid x) = \sigma(\text{margin})  
$$

Then define loss:

$$  
L = -\log \sigma(\text{margin})  
$$

Why the negative log?

Because:

- high probability should mean low loss

- logs turn products into sums

- sums are easier to optimize across datasets

- it avoids numerical underflow when multiplying many tiny probabilities

#### Intuition

If the model assigns:

- $0.99$ to the correct class, loss is tiny

- $0.5$ to the correct class, loss is moderate

- $0.01$ to the correct class, loss is huge

So logistic loss is not just asking “were you right?”  
It is asking:  
“How much probability did you put on being right?”

That is a richer learning signal.

---

### 12. Why logistic loss works for gradient descent

Unlike 0–1 loss, logistic loss is smooth.

That means:

- wrong examples produce a strong gradient

- barely correct examples still produce a useful gradient

- even confidently correct examples produce a small gradient pushing them to become even more confident

Percy’s nice implicit point here is that logistic loss is an “overachiever”:  
it does not stop caring the moment the example becomes correct.  
It still wants a healthier safety margin.

#### Intuition

Compare two students on a pass/fail exam:

- student A gets 51%

- student B gets 99%

0–1 loss says both are equally fine: they passed.  
Logistic loss says no: student B is much safer and more convincing.

That extra nuance is exactly what makes optimization possible.

---

### 13. Gradient descent on logistic loss

Now optimization works.

Procedure:

1. initialize parameters

2. compute training loss

3. compute gradient

4. update parameters

5. repeat

As training progresses:

- loss decreases

- decision boundary shifts

- training examples are classified correctly with increasing margin

A key insight is that the learned boundary does not just barely separate points. It tends to move to increase confidence as well.

---

### 14. Multi-class classification

Now the lecture generalizes from binary to multiclass.

#### Important terminology correction

What this lecture covers is **multiclass classification**, not **multi-label classification**.

- **multiclass**: exactly one class is correct  
    Example: image is cat, dog, or horse

- **multi-label**: several labels can be correct at once  
    Example: photo contains both beach and sunset


Lecture 3 is about **multiclass**.

#### Core extension

Binary classification:

- one score

Multiclass classification:

- one score per class

So if there are $K$ classes, the model computes:

$$  
z_0, z_1, \dots, z_{K-1}  
$$

Each class gets its own weight vector and bias.

---

### 15. Softmax: turning many scores into one distribution

Once you have multiple logits, you need probabilities that:

- are nonnegative

- sum to $1$

That is what softmax does:

$$  
P(y = c \mid x) = \frac{e^{z_c}}{\sum_j e^{z_j}}  
$$

#### Intuition-based analogy: horse race betting

Imagine each class enters a race:

- each class has a raw score $z_c$

- exponentiation turns each score into a positive “betting strength”

- normalization divides by the total betting strength

So the final probability for each class is:  
its share of the total evidence pool.

If one class has much larger logit, its exponentiated value dominates and it gets most of the probability mass.

#### Why exponentiate?

Exponentiation does two useful things:

- removes negative values

- exaggerates differences

A class with slightly larger logit becomes noticeably more probable.

That helps sharpen the distribution.

#### Why normalization?

Without normalization, you just have positive numbers, not probabilities.

Normalization forces the total to equal $1$, so the classes compete with each other.

This competition is important:  
giving more probability to one class automatically gives less to others.

---

### 16. Softmax as the multiclass analogue of the logistic function

This is the right mental link:

- logistic function: one binary logit $\to$ probability of positive class

- softmax: vector of class logits $\to$ probability distribution over all classes

So softmax is not a totally new idea. It is the multiclass generalization of the same probabilistic move we made in binary classification.

#### Strong analogy

Binary case:

- two sides of one fence

Multiclass case:

- several competing buckets

The model pours confidence into all buckets, and softmax rescales those amounts into a proper distribution.

---

### 17. Cross-entropy loss

Once softmax gives probabilities, we need a loss.

If true class is $t$, then:

$$  
L = -\log P(y = t \mid x)  
$$

That is the multiclass cross-entropy loss.

If you write the target as a one-hot vector, the same loss can be written as the cross-entropy between:

- target distribution

- predicted distribution

#### Intuition-based analogy: grading a multiple-choice answer sheet

Suppose the correct answer is class $2$.

If the model predicts:

- class $2$ with probability $0.9$, great

- class $2$ with probability $0.4$, not great

- class $2$ with probability $0.01$, terrible

Cross-entropy does not merely ask whether the correct class had the largest score.  
It asks:  
“How much confidence did you assign to the correct answer?”

That makes it the natural multiclass extension of logistic loss.

---

### 18. How multiclass extends binary classification

This is one of the places you said you were confused, so here is the clean bridge.

#### Binary classification

There are effectively two classes:

- negative

- positive

You can represent them with one logit because once you know the score for “positive,” the score for “negative” is determined by symmetry.

Then logistic converts that into probabilities.

#### Multiclass classification

Now there are $K$ classes, and there is no single “opposite” class.  
Each class needs its own score.

So the pipeline becomes:

1. compute one logit per class

2. compare all logits together

3. softmax converts them into probabilities

4. cross-entropy rewards high probability on the true class

#### Intuition-based analogy: choosing between two doors vs many doors

Binary classification is like deciding:

- left door or right door

One number is enough because positive means “more left” and negative means “more right.”

Multiclass classification is like choosing among $10$ doors.

Now a single left-right score is not enough.  
You need one score per door, then a mechanism to compare them all.

That mechanism is softmax.

So multiclass is not a tiny tweak. It is the natural scaling-up of the binary idea from:  
“which side?”  
to  
“which one among many?”

---

### 19. Representing text as tensors

The lecture then shifts to a practical question:

Machine learning uses tensors.  
Text is a string.  
So how do we turn text into tensors?

The answer is a pipeline:

1. tokenize the string

2. convert tokens to indices

3. optionally represent indices as one-hot vectors

---

### 20. Tokenization

Example:

`"the cat in the hat"`

A simple tokenizer splits on spaces:

`["the", "cat", "in", "the", "hat"]`

Then each unique token is assigned an integer:

- `"the"` $\to 0$

- `"cat"` $\to 1$

- `"in"` $\to 2$

- `"hat"` $\to 3$

So the sentence becomes:

$$  
[0, 1, 2, 0, 3]  
$$

Notice:

- repeated word `"the"` gets the same index both times

#### Intuition

Think of tokenization as creating a dictionary for the model.

Words are too symbolic for linear algebra.  
Indices are the model’s bookkeeping labels.

The model does not work directly with English words.  
It works with references into a vocabulary table.

#### Important nuance

Percy notes that splitting on spaces is naive.  
Modern language models use smarter tokenizers such as BPE.

That matters because:

- punctuation

- rare words

- subwords

- numbers

- hyphenation

- non-English morphology

all make naive word splitting brittle.

---

### 21. One-hot encoding

Once you have an index, you can represent it as a one-hot vector.

If vocabulary size is $4$ and token is `"hat"` with index $3$, then:

$$  
[0, 0, 0, 1]  
$$

So the sentence becomes a matrix:

- one row per position

- one-hot vector at each row

This gives a proper tensor representation.

#### Intuition-based analogy

Imagine a huge hotel with one room per vocabulary item.

A token’s one-hot vector is like a hallway of doors where exactly one room light is switched on.

That vector does not yet say anything semantic about the word.  
It only says:  
“the active word is room number 3.”

So one-hot encoding is an **identity marker**, not a meaning representation.

---

### 22. Operating directly on indices

The lecture also points out that in practice we often do not explicitly build the giant one-hot vectors.

Why?

Because they are sparse and wasteful.

Instead, indexing into a weight matrix using token IDs gives the same result much more efficiently.

Mathematically:

- one-hot vectors are nice for understanding

Computationally:

- indices are nicer for implementation

This distinction is important in ML:  
the clean mathematical object is not always the most efficient implementation.

---

### 23. Bag-of-words representation

This is the second area you wanted extra intuition on.

A bag-of-words representation takes all token vectors in a document and combines them, usually by summing or averaging, into one fixed-size vector.

So instead of preserving sequence positions, you collapse the document into:  
“which words appeared, and roughly how often?”

#### What it keeps

- vocabulary presence

- frequency information, if counts or averages are used

#### What it loses

- order

- syntax

- who did what to whom

- local phrase structure

This is why:

- “dog bites man”

- “man bites dog”

look identical under bag-of-words.

---

### 24. Intuition-based analogy for bag of words

#### Analogy 1: a shopping receipt

Imagine a sentence is a shopping trip and the words are items you bought.

A bag-of-words vector is like the final receipt:

- 2 apples

- 1 milk

- 3 bananas

The receipt tells you what was present and how much of it there was.

But it does **not** tell you:

- what order you picked items up

- what aisle you visited first

- what combinations appeared next to each other

So bag-of-words is a “contents summary,” not a “story of the sequence.”

That is exactly why it is useful but limited.

#### Analogy 2: a smoothie

A sentence is like a fruit bowl arranged in a particular order.

Bag-of-words throws all the fruit into a blender.

After blending, you can still tell what ingredients are present overall, but you can no longer recover:

- which fruit was next to which

- what came first

- what the structure looked like

The representation is simpler and fixed-size, but structural information is gone.

---

### 25. Why bag of words still matters

Even though bag-of-words is primitive compared to transformers, it is still pedagogically important because it shows a key ML pattern:

> turn a variable-length symbolic object into a fixed-size numeric vector

That is a huge idea.

Before deep learning, many practical NLP systems were built on variants of bag-of-words, tf-idf, n-grams, and other linear-feature methods.

So this is not just toy material. It is a foundational stepping stone.

---

### 26. How bag-of-words links back to linear classification

This is the conceptual connection tying the whole lecture together.

Earlier, the classifier assumed the input $x$ was already a vector.

Bag-of-words gives you exactly such a vector for text.

So the pipeline becomes:

1. start with text

2. tokenize it

3. convert it into a bag-of-words vector

4. feed that vector into a linear classifier

5. compute logits

6. use logistic or softmax loss depending on the task

That is how the “text representation” part fits into the earlier “linear classification” part.

Without this representation step, the earlier classifier formulas cannot directly handle text.

---

## Summary

This lecture extends the previous lecture’s learning framework from **real-valued prediction** to **discrete classification**.

The central ideas are:

- A **linear classifier** computes a logit $w \cdot x + b$.

- The **decision boundary** is where that logit is zero.

- The **margin** packages together correctness and confidence.

- **0–1 loss** reflects the true classification objective, but it is too discontinuous for gradient descent.

- To make optimization work, we switch from hard labels to **probabilities**.

- The **logistic function** converts a binary logit into a probability.

- **Logistic loss** is the negative log probability of the correct class and gives a smooth, optimizable objective.

- In the **multiclass** setting, we compute one logit per class.

- **Softmax** converts those logits into a probability distribution.

- **Cross-entropy loss** is the multiclass version of logistic loss.

- For text inputs, we need to convert strings into tensors using **tokenization** and **one-hot encoding**.

- A **bag-of-words** representation gives a simple fixed-dimensional vector for text, but sacrifices word order.

The deepest conceptual lesson of the lecture is this:

> Sometimes the objective we really care about is not directly easy to optimize, so we replace it with a smooth probabilistic surrogate that still points us in the right direction.

That idea will recur throughout machine learning.

---

## Real-World Applications

### 1. Spam detection

A bag-of-words vector of an email can be fed into a linear classifier to predict:

- spam

- not spam

Words like “free,” “winner,” or “urgent” may push the logit toward spam.

### 2. Sentiment analysis

A review can be tokenized and represented as bag-of-words or richer features, then classified as:

- positive

- negative

- neutral

This is one of the canonical uses of logistic and multiclass classification.

### 3. Medical diagnosis support

Given features from a patient record, a classifier can estimate probabilities for:

- disease A

- disease B

- disease C

This is a multiclass probabilistic prediction problem.

### 4. Image recognition

An image feature vector can be scored against multiple classes, then softmax can assign class probabilities like:

- cat

- dog

- horse

- airplane

### 5. Search and recommendation systems

A model may classify whether a user will:

- click

- not click

Binary logistic models are still widely used in ranking, recommendation, and advertising pipelines.

### 6. Foundation for later NLP models

Even though modern language models do much more than bag-of-words, the basic pipeline:

- tokenization

- numeric representation

- probabilistic prediction over tokens/classes

is foundational and reappears in much more advanced form later.