# CS221 Lecture 4 — Deep Learning (Conceptual Notes Only)

## Learning Objectives

By the end of this lecture, you should be able to:

* explain why linear models are limited
* explain how a nonlinear feature map can turn a nonlinear problem into a linear one in a new space
* explain why stacking only linear layers does **not** create a genuinely more expressive model
* explain why adding a nonlinear activation such as ReLU changes everything
* describe what a multi-layer perceptron (MLP) is conceptually
* explain the intuition behind deep neural networks as learned feature hierarchies
* explain the vanishing and exploding gradient problems
* explain why residual connections help optimization
* explain the purpose of layer normalization
* explain why initialization matters
* distinguish full gradient descent from stochastic gradient descent

---

## Concept Inventory

### 1. Linear predictor

A linear classifier or regressor computes a score from the input using a weighted sum.

For multiclass classification, the model produces one score per class, called a logit:

$$
s = Wx + b
$$

where:

* $x$ is the input vector
* $W$ is the weight matrix
* $b$ is the bias vector
* $s$ is the vector of logits

The key limitation is that this model can only carve the input space using straight boundaries.

---

### 2. Nonlinear feature map

A feature map transforms the original input into a new representation:

$$
\phi(x)
$$

Then a linear model is applied to that transformed input:

$$
s = W\phi(x) + b
$$

This is powerful because the classifier is linear in feature space, but nonlinear in the original input space.

---

### 3. Hidden layer

A hidden layer is an intermediate representation between input and output.

Conceptually, it is the model’s internal “working memory” or “intermediate description” of the data.

---

### 4. Activation function

An activation function applies a nonlinear transformation elementwise.

The lecture focuses on ReLU:

$$
\text{ReLU}(z) = \max(0, z)
$$

This keeps positive values and zeroes out negative ones.

---

### 5. Multi-layer perceptron (MLP)

An MLP is a neural network made of alternating linear transformations and nonlinear activations.

A simple two-layer MLP looks like:

$$
h = \text{ReLU}(W_1x + b_1)
$$

$$
s = W_2h + b_2
$$

where:

* $h$ is the hidden representation
* $s$ is the output logits

---

### 6. Deep neural network

A deep network stacks many such transformations:

$$
h_1 = \text{ReLU}(W_1x + b_1)
$$

$$
h_2 = \text{ReLU}(W_2h_1 + b_2)
$$

$$
\cdots
$$

$$
s = W_k h_{k-1} + b_k
$$

The intuition is that each layer may build more abstract features than the previous one.

---

### 7. Vanishing and exploding gradients

When gradients are multiplied through many layers, they can become:

* extremely small: vanishing gradients
* extremely large: exploding gradients

This makes deep networks hard to train.

---

### 8. Residual connection

A residual connection modifies a layer from:

$$
x \mapsto f(x)
$$

to:

$$
x \mapsto x + f(x)
$$

This creates a direct path for information and gradients to flow through the network.

---

### 9. Layer normalization

Layer normalization rescales activations so their magnitude stays under control.

In essence, it standardizes a vector by subtracting its mean and dividing by its standard deviation.

A simplified form is:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma}
$$

More complete versions include learned scale and shift parameters.

---

### 10. Initialization

Weights should start at sensible magnitudes. If they start too large or too small, the network may become unstable immediately.

A common idea is to scale weights by something like:

$$
\frac{1}{\sqrt{d_{\text{in}}}}
$$

where $d_{\text{in}}$ is the input dimension.

---

### 11. Stochastic gradient descent

Instead of computing the gradient over the whole dataset every step, we estimate it using a random mini-batch. This is cheaper and usually effective.

---

## Slide-by-Slide Walkthrough

## 1. From linear models to deep learning

The lecture begins by connecting deep learning to what you already saw:

* linear regression
* linear classification
* gradient descent
* backpropagation

So the lecture is **not** introducing a completely separate subject. It is extending the same machine-learning pipeline to more expressive predictors.

The big goal is:

> Move from linear predictors to nonlinear predictors.

That is the real conceptual jump.

A good way to think about it:

* previous lectures: “How do we fit a straight line or flat hyperplane?”
* this lecture: “How do we fit curved, layered, more flexible functions?”

---

## 2. PyTorch is not the concept — it is the machinery

Although much of the lecture uses PyTorch, the important conceptual point is this:

A deep-learning framework is just a tool that automates:

* forward computation
* gradient computation
* parameter updates

The real ideas are still:

* define a predictor
* define a loss
* compute gradients
* update parameters

So do not let the library distract you into thinking the mathematics changed. It did not. PyTorch is mainly a convenient engine for the same learning ideas.

A useful analogy:

* doing backprop by hand is like learning to build an engine from raw metal
* using PyTorch is like driving the finished car

You still need to understand what the car is doing, but you do not need to assemble every gear each time.

---

## 3. Computation graphs: values versus gradient flow

The lecture briefly revisits computation graphs.

The core idea is:

A quantity can play two roles:

* it has a numerical value
* it may also be part of a dependency chain for gradients

This matters because sometimes you want later computations to influence earlier parameters, and sometimes you do not.

Conceptually:

* if node $B$ is built from node $A$, then gradients can flow from $B$ back to $A$
* if you “detach” and only copy the value, then the number survives, but the dependency path does not

Analogy:

Imagine a company spreadsheet.

* If one cell directly references another cell, changing the source updates downstream calculations.
* If you manually copy the number into a new sheet, the number is the same, but the connection is broken.

That is the conceptual meaning of detaching:
same number, broken causal chain.

For your note on ML concepts, the key takeaway is not the syntax. It is this:

> Backpropagation only works along graph connections that were preserved.

---

## 4. Review of the training loop

The lecture next revisits the standard learning loop in deep-learning language.

Every training pipeline still has the same four stages:

### Step 1: Forward pass

Take inputs, produce predictions.

### Step 2: Loss computation

Measure how wrong the predictions are.

### Step 3: Backward pass

Compute gradients of the loss with respect to parameters.

### Step 4: Optimization step

Update parameters using those gradients.

This is the same logic as before. The main difference is that larger models now make these steps more expensive and more delicate.

A compact conceptual picture is:

$$
\text{input} \rightarrow \text{prediction} \rightarrow \text{loss} \rightarrow \text{gradient} \rightarrow \text{update}
$$

That pipeline is the backbone of almost all supervised deep learning.

---

## 5. Why linear classifiers are limited

This is the most important conceptual pivot in the lecture.

A linear classifier divides space with a straight boundary.

In two dimensions, that means a line.
In three dimensions, a plane.
In higher dimensions, a hyperplane.

That works well only when the classes are separable by a straight cut.

But many real datasets are not shaped that way.

For example:

* points inside a circle vs outside a circle
* curved decision boundaries
* patterns that require combinations of features

A line cannot represent those shapes well.

Analogy:

A linear classifier is like trying to cut a pizza into “good” and “bad” regions using exactly one straight knife cut. If the true pattern is ring-shaped, spiral-shaped, or made of several pockets, one cut is too simple.

So the question becomes:

> How do we keep the nice training machinery of linear models, but gain nonlinear expressive power?

---

## 6. The feature-map trick: nonlinear problem, linear solution in a new space

The lecture gives a very elegant solution:
transform the input first.

Suppose the original input is:

$$
x = (x_1, x_2)
$$

Now define a nonlinear feature map:

$$
\phi(x) = (x_1, x_2, x_1^2 + x_2^2)
$$

Then apply a linear classifier to $\phi(x)$.

This lets a linear model represent a circular decision boundary in the original space.

The lecture’s quadratic example is essentially:

$$
f(x_1, x_2) = (x_1 - 1)^2 + (x_2 - 1)^2 - 2
$$

Expanding gives:

$$
f(x_1, x_2) = x_1^2 + x_2^2 - 2x_1 - 2x_2
$$

This can be rewritten as a linear function of transformed features:

$$
f(x) = w \cdot \phi(x)
$$

with

$$
\phi(x) = (x_1, x_2, x_1^2 + x_2^2)
$$

and

$$
w = (-2, -2, 1)
$$

So although the classifier is nonlinear in the original coordinates, it is linear in the transformed coordinates.

This is a huge idea in machine learning.

Analogy:

Imagine you cannot solve a problem on a flat sheet of paper, so you fold the paper into a new shape. In the folded space, a straight line might now separate the data perfectly.

That is what a feature map does:
it changes the geometry of the problem.

---

## 7. Why fixed feature maps are useful but limited

This trick is powerful, but it creates a new problem:

Who designs the feature map?

If a human must manually invent $\phi(x)$, then learning is only doing half the job. You are still hand-engineering the representation.

This is where deep learning enters.

The deep-learning dream is:

> Do not just learn the final weights. Learn the feature map too.

So instead of:

1. human designs representation
2. learning finds weights

we want:

1. learning discovers representation
2. learning also finds weights

That is why hidden layers matter.

---

## 8. Two-layer network: feature map first, linear predictor second

The lecture then introduces a two-layer setup.

Conceptually:

* first layer: transforms input into hidden features
* second layer: turns hidden features into logits

So the first layer behaves like a learned feature map, and the second acts like a classifier on those learned features.

This is the correct intuition.

However, the lecture then makes an important correction:

> If both layers are only linear, the whole network is still just linear.

Mathematically,

$$
s = W_2(W_1x + b_1) + b_2
$$

which can be rearranged as

$$
s = (W_2W_1)x + (W_2b_1 + b_2)
$$

This is still just one linear transformation plus a bias.

So stacking linear layers without any nonlinearity is like stacking transparent sheets of glass: no matter how many you stack, you still see straight through.

This point is crucial.

A very common beginner mistake is:

> “More layers automatically means more power.”

No.
More layers only help if at least some layer introduces nonlinearity.

---

## 9. Why nonlinearity is the real turning point

To escape linearity, we need a nonlinear activation function.

This is the moment where the model becomes genuinely more expressive.

The lecture mentions several possibilities:

* sigmoid
* tanh
* ReLU
* swish
* GeLU

but focuses on ReLU:

$$
\text{ReLU}(z) = \max(0, z)
$$

ReLU does two things:

* negative values become $0$
* positive values stay unchanged

So it is partly linear and partly cut off.

Why is that useful?

Because it gives the network the ability to build piecewise nonlinear functions.

Analogy:

A linear model is like a single rigid ruler.
A ReLU network is like a structure made of hinged ruler segments. Each segment is still straight locally, but together they can bend into more complicated shapes.

That is why ReLU is so important:
it introduces bends.

---

## 10. Why not just use any nonlinearity?

The lecture also stresses that nonlinearities can create optimization problems.

The central tension is:

* stronger nonlinearity gives more expressive power
* but it can also damage gradients

This is a recurring deep-learning theme:
you want models expressive enough to represent complex functions, but not so unstable that training breaks.

ReLU works well partly because:

* for positive inputs, the gradient is $1$
* so it does not squash gradients there

That makes it easier to optimize than functions like sigmoid, which can become very flat.

---

## 11. Dead neurons and why zero gradients are dangerous

The lecture correctly warns you to be suspicious whenever gradients are zero or near zero.

For ReLU:

$$
\text{ReLU}(z) = 0 \quad \text{for } z \le 0
$$

and in that region the gradient is also zero.

So if a neuron always receives negative input, it always outputs zero and gets no learning signal. It becomes a “dead neuron”.

Analogy:

Imagine a worker in a factory whose conveyor belt is permanently shut off. Since no items ever reach them, they never contribute anything, and there is no feedback to improve their behavior. They are present in the system, but functionally inactive.

Why this matters:

* dead neurons waste capacity
* too many dead neurons can slow or damage learning

The lecture also makes an excellent broader point:

Near-zero gradients are often almost as bad as exactly-zero gradients.

That is why sigmoid can be problematic too. Its slope is not exactly zero in the extremes, but it can become so small that learning becomes painfully slow.

---

## 12. Multi-layer perceptron (MLP): the first genuinely nonlinear network

Once you insert ReLU between the layers, you get:

$$
h = \text{ReLU}(W_1x + b_1)
$$

$$
s = W_2h + b_2
$$

Now the model is not reducible to one linear map.

This is the first real neural network in the lecture.

The hidden vector $h$ is the learned feature representation.
The output logits $s$ are the class scores.

The key conceptual interpretation is:

> The model is learning how to represent the data before classifying it.

That is much more powerful than simply fitting a linear boundary in the original input space.

---

## 13. Why deeper networks might help

A single hidden layer can already be very expressive, but the lecture then asks:
what if one hidden layer is not enough?

The intuition given is:

Each layer can learn increasingly abstract features.

For images, a rough story is:

* early layers detect edges
* middle layers detect parts
* later layers detect objects

This should be treated as intuition, not as a theorem. But it is useful intuition.

Analogy:

Think of a manufacturing pipeline.

* raw metal enters the first station
* intermediate parts are made in the second
* assembled components are made in the third
* finished products appear at the end

Deep networks do something similar with information:
raw input is gradually reshaped into more task-relevant internal representations.

This is why the lecture frames deep learning as “learning the feature map”.

---

## 14. Why deep networks are hard to train

If depth gives so much power, why did people struggle with deep learning for so long?

Because training deep networks is fragile.

The lecture explains this with vanishing and exploding gradients.

Suppose each layer roughly multiplies by a number $w$.

After many layers, you effectively get something like:

$$
w^{20}
$$

or more generally $w^L$ for depth $L$.

Now:

* if $|w| < 1$, repeated multiplication shrinks toward zero
* if $|w| > 1$, repeated multiplication grows too large

So as information and gradients pass through many layers, they can either die out or blow up.

This is the right intuition.

Analogy:

Imagine whispering a message through 20 people.

* If each person speaks only half as loudly, the message fades away: vanishing.
* If each person shouts twice as loudly, the message becomes chaotic and distorted: exploding.

Deep networks are hard because signals are repeatedly transformed, and repeated transformations can destabilize magnitude.

---

## 15. Vanishing gradients

Vanishing gradients happen when backward signals become tiny.

Then parameter updates also become tiny.

That means:

* the network technically trains
* but in practice learns extremely slowly or not at all

Why especially harmful in deep networks?

Because early layers are far from the output.
If the gradient gets multiplied by many small factors, those earliest layers receive almost no signal.

So the model may fail to learn useful low-level features.

---

## 16. Exploding gradients

Exploding gradients are the opposite problem.

Gradients become huge, causing:

* unstable updates
* overshooting
* numerical overflow
* training divergence

In practical terms, the optimization process can become erratic or break entirely.

So the lesson is not merely “small bad, large bad”.

It is:

> Deep learning needs controlled signal magnitudes.

That theme connects directly to residual connections, normalization, and initialization.

---

## 17. Residual connections: letting information skip trouble spots

Residual connections are one of the most important ideas in modern deep learning.

Without a residual connection, a layer computes:

$$
x \mapsto f(x)
$$

With a residual connection, it computes:

$$
x \mapsto x + f(x)
$$

Why is this helpful?

Because the network no longer has to push all information through the complicated transformation $f$.
Some information can go straight through.

This helps in two ways:

### Information flow

The input is preserved more directly.

### Gradient flow

During backpropagation, gradients also have a direct route backward.

Analogy:

Suppose a city has many side roads with traffic lights and construction. A residual connection is like building an express bypass road that lets traffic continue even if the local roads are messy.

That is why the lecturer calls it an “escape hatch”.

Another useful intuition:
if the ideal thing for a layer to do is “almost nothing,” residual structure makes that easy. The layer can simply learn a small correction $f(x)$ instead of an entirely new transformation from scratch.

So residual networks are often easier to optimize because each layer learns a residual adjustment, not a full reinvention.

---

## 18. Why residual connections help with vanishing gradients

The lecture gives a scalar intuition:
if a layer behaves like multiplication by $w$, then repeated multiplication is fragile.

But with a residual structure, you get something behaving more like:

$$
x \mapsto (1 + w)x
$$

Now the number $1$ is doing important stabilizing work.

Why?

Because values near $1$ preserve magnitude much better than values near $0$ or very large values.

This is not a full proof of stability, but it is the correct intuition:
residual connections make it easier for the network to preserve and transmit useful signal.

---

## 19. Layer normalization: keep activations in a healthy range

The next idea is normalization.

The lecture’s core message is simple:

> We want activations to stay away from zero and infinity.

Layer normalization does this by standardizing the activation vector.

A simplified picture is:

1. compute mean
2. compute standard deviation
3. subtract mean
4. divide by standard deviation

So if the raw activations are too large, too small, or unevenly scaled, layer norm brings them back into a more controlled range.

Analogy:

Think of a classroom where one student is whispering and another is screaming. Before discussion can proceed sensibly, you normalize everyone’s volume so they are speaking at roughly comparable levels.

That is what layer norm does to activations.

This helps because:

* optimization becomes more stable
* the network is less likely to drift into bad numerical regimes
* training deeper models becomes easier

The lecture also notes that real layer norm includes learned scale and shift parameters, so the model can still choose an appropriate representation rather than being rigidly standardized forever.

---

## 20. Initialization: start in a region where learning is possible

Initialization matters because the very first forward and backward passes already determine whether training begins in a sane regime.

If weights start too large:

* activations can become huge
* gradients can explode

If weights start too small:

* activations can shrink
* gradients can vanish

So the goal is to initialize weights in a scale-aware way.

The lecture gives the main idea:

scale weights by roughly

$$
\frac{1}{\sqrt{d_{\text{in}}}}
$$

where $d_{\text{in}}$ is the number of incoming inputs.

Why does this help?

Because when you take a dot product of many random terms, the magnitude tends to grow with dimension. Dividing by $\sqrt{d_{\text{in}}}$ counteracts that growth.

Analogy:

If 16,000 people each add a small random shove to a cart, the total motion can become large just because there are so many contributors. Scaling by $\sqrt{d_{\text{in}}}$ is like reducing the force each person contributes so the total remains manageable.

This is the conceptual role of Xavier-style initialization:
start the network in balance.

---

## 21. Gradient descent versus stochastic gradient descent

The lecture ends with optimization at dataset scale.

### Full gradient descent

Use the whole dataset to compute one exact gradient step.

Pros:

* accurate gradient

Cons:

* expensive for large datasets

### Stochastic gradient descent

Use a random subset, called a mini-batch, to estimate the gradient.

Pros:

* much cheaper per step
* faster in practice
* standard for deep learning

Cons:

* noisier gradient estimate

The key idea is that a mini-batch gradient is an unbiased estimate of the full gradient, provided the batch is sampled appropriately.

Analogy:

Suppose you want to estimate average public opinion in a country.

* full gradient descent = interview every person
* stochastic gradient descent = interview a well-chosen random sample

The sample is noisier, but vastly cheaper.

That is why SGD and its variants dominate deep learning.

---

## 22. Adam and “better optimizers”

The lecture only touches this briefly, but the conceptual message is:

Different optimizers use gradients differently.

SGD uses the current gradient fairly directly.
Adam adds adaptive scaling and momentum-like behavior, often making training faster and more stable in practice.

You do not need the full Adam equations yet to understand the role:
it is another tool for making optimization behave well.

---

## 23. The deepest conceptual thread of the lecture

If you zoom out, Lecture 4 is really about one theme:

> More expressive models are also harder to optimize.

That is the central trade-off.

Deep learning gives us:

* learned feature maps
* nonlinear decision boundaries
* representation hierarchies
* enormous expressive power

But then we must fight:

* dead neurons
* vanishing gradients
* exploding gradients
* unstable activations
* poor initialization
* expensive optimization

So deep learning is not just “bigger models”.
It is a careful engineering of expressivity plus trainability.

That is the intellectual heart of the lecture.

---

## Summary

Lecture 4 extends earlier ideas from linear regression/classification into the world of deep learning.

The path of ideas is:

1. linear models are limited because they only make straight cuts through input space
2. one workaround is a nonlinear feature map followed by a linear predictor
3. the deeper idea is to learn that feature map automatically
4. stacking only linear layers does not help, because linear compositions remain linear
5. nonlinear activations such as ReLU make networks genuinely more expressive
6. hidden layers can be interpreted as learned internal features
7. deeper networks may learn increasingly abstract representations
8. but deep networks are hard to optimize because of vanishing and exploding gradients
9. residual connections help gradients flow
10. layer normalization keeps activations in a controlled range
11. proper initialization prevents bad scales from the start
12. stochastic optimization makes learning feasible on large datasets

The cleanest one-sentence takeaway is:

> Deep learning is the art of learning powerful feature representations while keeping optimization stable enough for gradient-based learning to work.

---

## Real-World Applications

### Computer vision

Deep networks can learn layers of visual abstraction:
edges $\rightarrow$ parts $\rightarrow$ objects.

Examples:

* image classification
* object detection
* medical image analysis

### Natural language processing

Neural networks learn distributed representations of words, phrases, and sentences rather than relying entirely on hand-designed language features.

Examples:

* translation
* summarization
* question answering
* chatbots

### Speech

Deep models learn intermediate acoustic patterns and higher-level phonetic or linguistic structure.

Examples:

* speech recognition
* speaker identification

### Recommendation systems

Neural networks can learn complex nonlinear interactions between:

* users
* items
* context

### Scientific and business forecasting

Nonlinear models can capture richer patterns than linear baselines when the relationship between variables is complex.

Examples:

* demand forecasting
* anomaly detection
* biological prediction tasks

### Why the lecture matters beyond theory

This lecture explains why modern AI systems are not just “linear models with more data.”
They rely on:

* learned internal representations
* nonlinear transformations
* stabilization tricks that make deep training actually possible

Without those ideas, most modern large-scale AI would be much weaker or impossible to train effectively.


