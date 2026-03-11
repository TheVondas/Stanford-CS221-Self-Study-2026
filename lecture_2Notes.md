This note is based on the lecture transcript together with the course source files `backpropagation.py` and `linear_regression.py` from the Autumn 2025 lecture repository.

## Learning Objectives

By the end of this lecture, I should be able to:

- explain what a tensor is and what Percy means by tensors being the “atoms” of modern machine learning;
- distinguish tensor order from matrix rank, and explain axes clearly;
- read basic `einsum` expressions by tracking which named axes survive and which are summed out;
- understand `einsum` as a bookkeeping language for additions and multiplications over tensor axes;
- explain why `einops` / named-axis notation is often easier to reason about than raw transpose-based code;
- define an objective function as a mapping from tensor inputs to a scalar output;
- interpret a gradient as “if I nudge the input slightly, how much does the objective change?”;
- explain why gradients have the same shape as the tensor they differentiate with respect to;
- describe a computation graph in terms of nodes, dependencies, values, and gradients;
- explain backpropagation as a forward pass plus backward pass organized by the chain rule;
- define the supervised learning pipeline for linear regression: predictor, data, hypothesis class, loss, optimization;
- derive the intuition behind residuals, squared loss, training loss, and gradient descent;
- explain why gradient descent moves opposite the gradient and how the learning rate affects convergence.

## Concept Inventory

### 1) Tensors

A tensor is just a multi-dimensional array.

- Order 0 tensor: scalar
- Order 1 tensor: vector
- Order 2 tensor: matrix
- Higher-order tensors: arrays with 3 or more axes

Percy prefers the word **order** instead of **rank** here, because matrix rank already means something different in linear algebra.

A tensor’s order is simply the number of axes it has.

Examples:

- scalar → no axes
- vector → one axis
- matrix → two axes
- image batch → often four axes, such as batch × height × width × channel

In ML, tensors represent almost everything:

- inputs
- targets
- parameters
- gradients
- intermediate computations
- activations

This is why Percy says tensors are the atoms of modern machine learning.

### 2) Axes and named axes

An axis is just one “direction” along which a tensor varies.

For a matrix:

- axis 0 = rows
- axis 1 = columns

The key `einops` idea is that instead of thinking only in terms of axis numbers like `0`, `1`, `-1`, `-2`, you name the axes according to what they mean.

Examples:

- `example feature`
- `batch seq hidden`
- `height width channel`

This is like choosing variable names in code:

- the computer does not care much what you call them;
- the human absolutely does.

Good axis names reduce confusion.

### 3) Core intuition for `einsum`

`einsum` is easiest to understand as a **routing and accumulation rule**.

Percy’s informal rule is:

- for every assignment of the input axes;
- index into each input tensor;
- multiply the selected values;
- add the result into the output location determined by the output axes.

That sounds abstract at first, but it becomes much simpler if you use this checklist:

1. Look at the axes on the left.
2. Look at the axes on the right.
3. Any axis that disappears is being summed over.
4. Any axis that remains defines the shape of the output.

So `einsum` is really “multiply, then sum out the missing labels.”

### 4) Layman’s analogy for `einsum`: labeled shelves

Imagine a warehouse with labeled shelves.

- one label might be `i`
- another might be `j`
- another might be `k`

Each tensor tells you how items are stored on shelves with certain labels.

The `einsum` string tells you:

- which labels to line up;
- which labels to keep in the final answer;
- which labels to collapse by adding over them.

So instead of memorizing formulas, you are just telling the computer how to match labels and where to pour the results.

### 5) Another analogy: `einsum` as a recipe contract

Left-hand side = ingredients and their labels.

Right-hand side = the form of the final dish.

If a label is present on the left but absent on the right, it gets “used up” during cooking. In `einsum`, “used up” means summed out.

This is why matrix multiplication, dot product, row sums, transposes, and outer products all fit under the same umbrella.

### 6) Why Percy likes this better than transposes

Transpose-heavy code often forces you to reason like this:

- which axis was `-1` again?
- did I flip rows/columns correctly?
- is this the sequence axis or the hidden axis?

Named-axis notation lets you reason from meaning instead:

- `i j -> j i` means transpose;
- `i j, j -> i` means matrix-vector product;
- `i, j -> i j` means outer product.

That is more readable because the semantics are in the pattern itself.

### 7) Basic `einsum` patterns from the lecture

#### Identity

`i -> i`

Interpretation:

- for each `i`, send `x[i]` to `y[i]`
- output has the same axis as the input

#### Sum all entries of a vector

`i ->`

Interpretation:

- input depends on `i`
- output has no axes, so it is a scalar
- therefore sum over all `i`

#### Elementwise product

`i, i -> i`

Interpretation:

- match the same index `i` across both vectors
- multiply element-by-element
- keep `i` in the output

#### Dot product

`i, i ->`

Interpretation:

- match by `i`
- multiply element-by-element
- sum out `i`
- output is a scalar

#### Outer product

`i, j -> i j`

Interpretation:

- do not match the indices, because `i` and `j` are different
- keep both axes
- output is a matrix

This is why outer product creates something larger instead of collapsing dimensions.

### 8) Matrix examples from the lecture

#### Sum all entries of a matrix

`i j ->`

Everything is summed into one scalar.

#### Row sums

`i j -> i`

- keep row index `i`
- sum out column index `j`

#### Column sums

`i j -> j`

- keep column index `j`
- sum out row index `i`

#### Transpose

`i j -> j i`

You simply swap the labels.

#### Matrix-vector product

`i j, j -> i`

This is one of the most important patterns to get comfortable with.

Interpretation:

- matrix has axes `i j`
- vector has axis `j`
- match on `j`
- keep `i`

So you multiply row `i` of the matrix against the vector and sum over `j`.

#### Matrix-matrix product

`i k, j k -> i j`

Interpretation:

- match on `k`
- keep `i` and `j`
- sum out `k`

The exact pattern depends on which product you want, but the reading rule is always the same.

### 9) The `+=` intuition Percy emphasizes

Percy writes the conceptual update as `y[...] += ...`.

Why `+=`?

Because many different assignments of the input indices can contribute to the same output cell.

If an input axis disappears from the output, that means multiple terms are being accumulated into one place.

So `+=` is there to remind you that `einsum` is often doing repeated accumulation, not just one assignment.

### 10) Objective functions

The lecture then shifts from tensor mechanics to learning.

An **objective function** is a function that takes some tensor input and returns a scalar.

Why scalar?

Because if you want to optimize something, you usually want one number telling you how good or bad the current setting is.

Motivating linear-regression-style example:

- data matrix `X`
- targets `y`
- weight vector `w`
- predictions = `Xw`
- residuals = predictions − targets
- losses = residuals squared elementwise
- total loss = sum of losses

So one vector `w` gives one scalar loss.

That scalar is what optimization will try to reduce.

### 11) Gradient intuition

The gradient is one of the central ideas of the lecture.

The informal meaning is:

- if I nudge the input a tiny bit, how much does the output change?

In 1D, this is just the derivative.

In multiple dimensions, you have one partial derivative per input component, and you stack them together into a vector or tensor.

So the gradient is the complete local sensitivity map.

### 12) Layman’s analogy for gradients: fog on a hill

Imagine standing on a hill in thick fog.

You cannot see the whole landscape.

But you can feel the slope directly under your feet.

- the gradient points in the direction of steepest uphill climb;
- the negative gradient points in the direction of steepest downhill descent.

This is the intuition behind gradient descent.

### 13) Why the gradient has the same shape as the input

If a function takes in a tensor and returns a scalar, then the gradient tells you the sensitivity of that scalar with respect to each input entry.

So every input entry needs its own derivative.

Therefore:

- input shape = gradient shape

This is a very important sanity check.

If your input is a `2 × 3 × 4` tensor, then the gradient with respect to that input is also `2 × 3 × 4`.

### 14) Backpropagation motivation

In principle, you could differentiate every function by hand.

In practice, that becomes tedious and error-prone very quickly.

But most ML objectives are built from a small set of primitive operations:

- addition
- subtraction
- multiplication
- exponentials
- logs
- powers

So instead of redoing calculus from scratch every time, we can represent the computation as a graph and apply a general algorithm.

That algorithm is backpropagation.

### 15) Computation graphs

A computation graph is a graph where:

- leaf nodes are fixed inputs;
- internal nodes are operations applied to earlier nodes;
- the root is the final output.

For the simple example

`y = (x1 + x2)^2`

the graph is:

- `x1`
- `x2`
- `sum = x1 + x2`
- `y = sum^2`

Each node stores:

- a name
- dependencies
- value
- gradient

### 16) Forward pass

The forward pass computes the values.

For example:

- `x1 = 2`
- `x2 = 3`
- `sum = 5`
- `y = 25`

So the forward pass answers:

“What is the output?”

### 17) Backward pass

The backward pass computes gradients.

The central interpretation Percy gives is:

- a node’s gradient means: if I change this node’s value by a tiny amount, how much does the root output change?

This is extremely important.

It means that every gradient is always understood **with respect to the final output at the root**.

### 18) Why the root gradient starts at 1

At the root, the question is:

- if I change the root by ε, how much does the root change?

Answer: exactly ε.

So the derivative of the root with respect to itself is 1.

That is why backprop starts by setting:

- root gradient = 1

This is one of the most common conceptual sticking points, so remember it well.

### 19) Backprop as blame assignment

A useful analogy:

- forward pass computes the final score;
- backward pass assigns blame or credit backward through the graph.

If the final output changed, how much responsibility does each earlier node bear?

Each backward step pushes this influence signal to the dependencies.

### 20) Chain rule in graph form

Backprop is just the chain rule, organized systematically.

For a node `c` depending on `b`, which depends on `a`:

`dc/da = (dc/db)(db/da)`

So if you know:

- how changing `b` affects the output;
- how changing `a` affects `b`;

then you can combine them.

That is exactly what the backward methods are doing.

### 21) Why topological order matters

Percy emphasizes topological sort because the object is a graph, not necessarily a tree.

A node may feed into multiple later nodes.

So we need to:

- compute forward values only after dependencies are ready;
- compute backward gradients only after downstream gradients are ready.

Topological ordering guarantees this bookkeeping is correct.

### 22) Linear regression setup

The lecture then moves into supervised learning.

Example task:

- input: number of hours studied
- output: exam score

A predictor is any function mapping input to output.

Example fixed predictor:

`f(x) = 2x + 1`

But machine learning is not about hand-picking one fixed predictor.

It is about defining a family of possible predictors and letting data choose among them.

### 23) Training data and the learning problem

Training data is a set of examples of the form:

- input
- output

In the lecture’s 1D example:

- `(1, 4)`
- `(2, 6)`
- `(4, 7)`

A learning algorithm takes this training data and returns a predictor.

Percy frames the whole problem around three questions:

1. Which predictors are possible?
2. How good is a predictor?
3. How do we find the best one?

Those correspond to:

- hypothesis class
- loss function
- optimization algorithm

### 24) Hypothesis class

For linear regression, the predictor has the form:

`ŷ = wx + b`

where:

- `w` = weight
- `b` = bias

A specific pair `(w, b)` gives one specific line.

The **hypothesis class** is the set of all such lines you can get by varying `(w, b)`.

Important conceptual distinction:

- hypothesis class = the family of allowed predictors
- model/predictor = one specific member of that family after choosing parameters

In deep learning language:

- hypothesis class ≈ architecture family
- one learned predictor ≈ one trained model

### 25) Weight and bias intuition

For `ŷ = wx + b`:

- `w` controls the slope
- `b` controls the vertical intercept

Analogy:

- `w` tells you how steeply the line tilts
- `b` tells you where the line crosses the vertical axis

### 26) Residuals and squared loss

For one example `(x, y)`:

- prediction = `wx + b`
- residual = prediction − true output
- loss = residual²

Residual tells you the signed error:

- positive residual → predicted too high
- negative residual → predicted too low
- zero residual → exact hit

Squaring makes all errors positive and punishes large mistakes more strongly.

### 27) Training loss

The training loss is the average of the per-example losses.

So if there are `N` examples:

`L(w, b) = (1/N) Σ (wx_i + b − y_i)^2`

This converts “fit all the points well” into one scalar objective.

The learning goal becomes:

find parameters `(w, b)` that minimize training loss.

### 28) Gradient of the loss

The lecture gives the per-example gradient as:

`2 residual [x, 1]`

This is worth interpreting carefully.

- `residual` tells you the direction of error
- the factor `2` comes from differentiating the square
- `[x, 1]` tells you how changes in weight and bias affect the prediction

This gives a two-dimensional gradient because the parameter vector here has two entries:

- weight
- bias

The gradient for the whole training set is just the average of the per-example gradients.

### 29) Why gradient descent works here

Gradient tells you the direction of steepest increase.

So to reduce loss, move in the opposite direction.

Update rule:

- `w ← w − η (∂L/∂w)`
- `b ← b − η (∂L/∂b)`

where `η` is the learning rate.

This is gradient descent.

### 30) Learning rate intuition

Percy’s driving analogy is exactly right and worth remembering.

- too small → progress is safe but slow
- too large → you overshoot or diverge
- moderate → steady progress

So the learning rate is not just a technical detail. It is the knob that controls the stability-speed tradeoff.

### 31) Convergence and convexity

For linear regression with squared loss, the objective is convex.

That means the landscape is bowl-shaped rather than full of deceptive local traps.

So gradient descent is much better behaved here than it typically is in deep learning.

This is why linear regression is such a good teaching example: the mechanics are the same as more advanced ML, but the optimization story is cleaner.

### 32) Big picture of the lecture

This lecture really has one master arc:

1. Represent everything with tensors.
2. Express computations cleanly with named axes.
3. Turn computations into scalar objectives.
4. Use gradients to understand local improvement.
5. Use computation graphs and backprop to compute those gradients efficiently.
6. Use gradient descent to learn parameters from data.

That is a huge fraction of modern machine learning in miniature.

## Slide-by-Slide Walkthrough

### 1. Review of tensors and why they matter

Percy opens by reconnecting the lecture to the previous one: tensors are the base objects of modern ML. This is not just terminology. In ML, you almost never think in one isolated number at a time. You think in vectors, matrices, batches, and higher-dimensional arrays.

### 2. Why revisit `einops`

He explicitly slows down because `einops` is unfamiliar to most people at first. That is a good signal: if it feels strange, that is normal. The real goal is not to memorize syntax, but to internalize the reading process.

### 3. Tensor order and axes

He defines tensor order carefully so you do not confuse it with matrix rank. Once you are comfortable with “order = number of axes,” a lot of shape reasoning becomes easier.

### 4. Naming axes

The key `einops` mindset is semantic naming. The name should reflect what the axis means. If rows are examples and columns are features, call them `example feature`. You are encoding meaning, not merely shape.

### 5. `einsum` identity and sum

Percy starts with trivial examples on purpose. `i -> i` and `i ->` look almost silly, but they establish the two most important ideas:

- the right-hand side determines the output shape;
- missing axes are summed away.

### 6. Elementwise product and dot product

`i, i -> i` and `i, i ->` differ only by what survives on the right-hand side. That shows how one small pattern change turns elementwise multiplication into dot product.

### 7. Outer product

`i, j -> i j` introduces the idea that if labels do not match, they do not collapse. Instead, they combine into a larger output. This is where `einsum` starts to feel like a general tensor language rather than one isolated trick.

### 8. Matrix examples

The matrix section is where the notation starts to feel genuinely useful. Row sums, column sums, transpose, matrix-vector product, and matrix-matrix product all become variations of the same label game.

### 9. General `einsum` rule

Percy’s summary rule is the one to remember: for each assignment of the input axes, multiply the relevant tensor entries and add the result into the output entry indexed by the output axes.

### 10. Shift to objective functions

The lecture then pivots from representation to optimization. Once you know how to build computations out of tensors, the next question is how to improve those computations.

### 11. Linear-regression tensor mechanics

The first motivating example is deliberately mechanical: `Xw`, residuals, squared losses, sum. Percy is telling you not to worry yet about the statistical meaning. He wants you to see how a learning objective is built out of ordinary tensor operations.

### 12. Objective functions as scalar-valued functions

A whole chain of tensor operations gets wrapped into one scalar-valued function of the parameters. That scalar is what optimization sees.

### 13. 1D derivative refresher

The `x^2` example reconnects geometric intuition to calculus. The derivative is the local rate of change, or the slope of the tangent line.

### 14. 2D gradient refresher

For `f(x1, x2) = (x1 + x2)^2`, Percy shows that the gradient gives the direction of steepest increase. This is the bridge from single-variable calculus to parameter optimization.

### 15. Vector-valued input case

The vector example generalizes the idea further. The gradient becomes a tensor of the same shape as the input. This is the right mental model before entering backpropagation.

### 16. Why manual differentiation does not scale

Even if the primitive operations are simple, the overall function may be large and composed. That is why a systematic computational method is needed.

### 17. Computation graph construction

The graph viewpoint externalizes the function. Instead of seeing one complicated expression, you see a network of dependencies. This is the right preparation for understanding autodiff libraries later.

### 18. Forward pass

The forward pass computes values from leaves to root. Conceptually, it is just evaluation.

### 19. Backward pass and root gradient

The backward pass computes gradients from root to leaves. The key anchor is that the root gradient is 1 because the output changes one-for-one with itself.

### 20. Local meaning of each node’s gradient

Each node’s gradient tells you how much the final output would change if you nudged that node’s value. That makes the whole backward pass much more concrete.

### 21. Backpropagation as chain-rule automation

Every backward method is just a local derivative rule plus the already-accumulated downstream gradient. That is chain rule in operational form.

### 22. Topological traversal

Backprop is not magic recursion. It is an ordered procedure: sort the graph, run forward, initialize gradients, then traverse backward.

### 23. Transition to supervised learning

After building gradient machinery in a generic setting, Percy narrows back to machine learning. First understand the tool, then see how it solves a learning problem.

### 24. Predictor and task

He defines a predictor as a function from input to output. The “hours studied → exam score” example is intentionally simple so that the ML template is not hidden by domain complexity.

### 25. Training data

The training examples make the task concrete. They are demonstrations of intended behavior.

### 26. Three-part ML template

One of the most important conceptual moments of the lecture is reducing the learning problem to three design choices:

- which predictors are allowed;
- how predictors are scored;
- how good predictors are found.

### 27. Hypothesis class

Instead of hard-coding one predictor, we define a family. That family is the hypothesis class. Learning means selecting good parameters from within that family.

### 28. Loss function

Residual measures miss distance. Squared loss turns miss distance into a scalar penalty. Averaging over examples gives training loss.

### 29. Optimization problem

Once you have training loss as a function of parameters, learning becomes optimization.

### 30. Gradient descent step

Percy computes the gradient, takes a small step opposite it, and observes the loss go down. This is the first complete learning loop in the course.

### 31. Iterating gradient descent

Repeating the update shows the loss steadily dropping. Even though the example is tiny, this is the same basic pattern used to train much larger models.

### 32. Learning rate and convexity note

He closes with the driving analogy and the convexity caveat. That distinguishes the clean linear-regression world from the messier deep-learning world you will encounter later.

## Summary

This lecture has four tightly connected themes.

First, it reviews tensors and named-axis reasoning. The message is that good tensor programming is not only about getting shapes to line up. It is about making the meaning of each axis explicit.

Second, it reframes `einsum` as one master tensor operation. You can read it by asking:

- which labels are present in the inputs?
- which labels survive into the output?
- which labels disappear and therefore get summed out?

If you can answer those questions, you can usually decode the operation.

Third, it develops gradient intuition. A gradient is the tensor of local sensitivities of a scalar objective with respect to each input component. It has the same shape as the input and points in the direction of steepest increase.

Fourth, it explains how gradients are computed and then used for learning. Computation graphs represent complex functions as compositions of primitive operations. Backpropagation applies the chain rule systematically over that graph. Linear regression then gives the first full machine-learning example where parameters, loss, gradient, and optimization all come together.

The deepest takeaway from this lecture is the overall pattern:

- build a parameterized computation;
- convert goodness into a scalar objective;
- compute gradients;
- update parameters to reduce the objective.

That pattern is the backbone of a large part of modern AI.

## Real-World Applications

### 1. Transformers and attention

Named axes like `batch`, `seq`, and `hidden` are exactly the kinds of dimensions that appear in attention models. `einsum`-style thinking helps you read and debug those architectures without getting lost in raw transpose operations.

### 2. Computer vision and multi-dimensional data

Images, videos, and batched sensor data are all tensors. `einops`-style notation is especially useful for reshaping and reasoning about these structures.

### 3. Automatic differentiation libraries

PyTorch, JAX, and related tools rely on the same underlying idea as the lecture’s mini backprop system: represent the computation, run it forward, then propagate gradients backward through dependencies.

### 4. Adversarial examples and input optimization

Percy briefly mentions adversarial examples. This is a strong reminder that gradients can optimize not just model parameters but also inputs, prompts, perturbations, data mixtures, and many other objects.

### 5. Classical regression and forecasting

Linear regression remains useful as:

- a baseline model;
- an interpretable model;
- a debugging tool;
- a first-pass approximation before trying something more complex.

### 6. Deep learning more broadly

Neural networks replace the simple line `wx + b` with much richer nonlinear functions, but the structure of the learning problem is the same:

- choose hypothesis class;
- define loss;
- compute gradients;
- optimize.

So this lecture is not just about linear regression. It is the miniature version of a much bigger story.

