## Learning Objectives

By the end of this lecture, you should be able to:

- explain Percy Liang’s working definition of AI in terms of four ingredients: perception, reasoning, action, and learning
- explain why intelligence must always be studied under resource constraints such as limited computation and limited information
- distinguish between developer goals and broader societal goals for AI systems
- describe the three major historical traditions in AI: symbolic AI, neural AI, and statistical AI
- explain why the course is “tensor-native” and why tensors are not just for deep learning
- interpret tensor rank, shape, slicing, broadcasting, and batched matrix multiplication in Python
- understand why vectorized tensor operations are preferred over manual loops
- explain the practical role of `einops` / named-dimension tensor notation for keeping high-dimensional computations readable

## Concept Inventory

### 1) What AI is, in this course

Percy’s framing is deliberately operational rather than philosophical. Instead of asking whether a machine is “really intelligent” in some abstract sense, he asks what an intelligent agent must be able to do.

The four ingredients are:

- **Perceive** the world
- **Reason** with what it knows
- **Act** in the world
- **Learn** from experience

That is a very useful framing because it turns “AI” from a vague buzzword into a set of concrete computational problems.

### 2) Resource constraints are part of intelligence

An agent is not intelligent in a vacuum. It must operate under:

- **computation constraints**: time, memory, communication
- **information constraints**: incomplete observations, limited data, limited experience

This is one of the deepest themes of the course. AI is not just “find the perfect answer.” It is “find the best possible answer given limited time, limited data, and an imperfect view of reality.”

### 3) Goals and alignment

AI systems do not pursue objectives magically. They are built by developers who encode goals, values, objectives, or utility functions either explicitly or implicitly.

Two different questions arise:

- **Developer-level question:** what does the builder want the system to do?
- **Society-level question:** what effects do we want these systems to have on the world?

This distinction matters because a system can be well-aligned with the developer’s goals and still be harmful socially.

### 4) Three traditions of AI

Percy organizes the history of AI into three overlapping traditions:

- **Symbolic AI**: logic, search, explicit rules, knowledge representation
- **Neural AI**: artificial neural networks, representation learning, deep learning
- **Statistical AI**: probability, optimization, inference, generalization

A major point of the lecture is that modern AI is not “one of these.” It is a melting pot of all three.

### 5) Tensors as the computational language of the course

A tensor is a multi-dimensional array.

Examples:

- scalar: rank 0
- vector: rank 1
- matrix: rank 2
- higher-order tensor: rank 3 or more

In modern ML, tensors represent almost everything:

- data
- model parameters
- activations
- gradients
- masks
- outputs

This is why Percy says tensors are the “atoms” of modern machine learning.

### 6) Why tensors matter beyond deep learning

A very important idea from the lecture is that tensor notation is not just a deep learning convenience.

Percy’s claim is stronger:

- value iteration can be expressed in tensor operations
- Bayesian network inference can be expressed in tensor operations
- many core AI computations can be phrased as tensor computations

So “tensor-native” means the course is adopting a general computational language, not merely following modern fashion.

### 7) Shapes are the grammar of tensor thinking

The most important habit to build early is: always know the shape.

Examples:

- a single data point: $x \in \mathbb{R}^{D}$
- a batch of data points: $X \in \mathbb{R}^{N \times D}$
- a batch of token sequences: $X \in \mathbb{R}^{N \times L \times D}$
- a batch of images: $X \in \mathbb{R}^{N \times H \times W \times C}$
- a weight matrix: $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$

If you understand the shape, you usually understand the computation.

### 8) Rank means number of axes here, not matrix rank

A common beginner confusion:

- in linear algebra, “rank” often means the dimension of a matrix’s column space
- in tensor programming, “rank” usually means the number of axes / dimensions

So a rank-3 tensor means it has 3 axes, not that some matrix has algebraic rank 3.

### 9) Broadcasting and batching

Broadcasting means a lower-rank tensor is implicitly reused across extra dimensions of a higher-rank tensor.

Typical example:

- if $X \in \mathbb{R}^{B \times N \times D_{\text{in}}}$
- and $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$

then $XW$ produces a tensor in $\mathbb{R}^{B \times N \times D_{\text{out}}}$ by applying the same matrix $W$ across the batch dimension.

This is one reason tensor code is compact and fast.

### 10) Vectorization beats manual loops

Percy compares a Python triple loop for matrix multiplication with NumPy matrix multiplication.

The lesson is not just “NumPy is convenient.”

The deeper lesson is:

- tensor libraries call highly optimized low-level code
- tensor operations map well to GPUs
- vectorized operations can be parallelized and distributed more effectively than hand-written Python loops

So tensorization is both a notation choice and a systems-performance choice.

### 11) Named dimensions reduce confusion

High-dimensional tensor code becomes unreadable when dimensions are tracked only by position, such as `-2` and `-1`.

This is why Percy introduces `einops` / Einstein-style notation.

Instead of thinking:

- “transpose the last two axes and hope I got them right”

you think:

- “this tensor has axes `batch seq hidden`”
- “I want `batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2`”

This is much closer to mathematical reasoning and much safer.

---

## Slide-by-Slide Walkthrough

### 1. Welcome and framing the course

The lecture opens by noting that AI has changed dramatically, but many foundational principles have not. That is the central philosophy of the course:

- the examples are modern
- the mathematical and algorithmic foundations are timeless

This is important because it tells you how to study the course. Do not treat it as a list of trendy tools. Treat it as a foundations course that happens to live in a modern AI era.

### 2. What is AI?

Percy starts with visible examples:

- AI assistants
- autonomous vehicles
- game-playing systems
- competition-level math/programming models
- scientific systems like AlphaFold

But he quickly moves away from examples and toward a definition.

His decomposition of intelligence is:

- **Perception**: turning raw sensory input into usable internal representations
- **Reasoning**: drawing conclusions and making decisions from information
- **Action**: producing outputs that affect the world
- **Learning**: improving from experience over time

This is a powerful decomposition because nearly every AI topic can be placed into one or more of these boxes.

#### Intuition

Think of an AI agent like a student learning to play football:

- perception: seeing where teammates and opponents are
- reasoning: deciding the best pass
- action: kicking the ball
- learning: improving after reviewing mistakes

If one of the four is missing, the agent is severely limited. A system that perceives and reasons but never acts cannot demonstrate competence. A system that acts but never learns remains brittle.

### 3. Example: autonomous driving

Autonomous driving is used to make the four-part decomposition concrete.

- perception: interpret camera, lidar, radar, and other sensor streams
- reasoning: infer what nearby agents will do and choose a plan
- action: accelerate, brake, steer
- learning: improve future behavior from accumulated experience

This example matters because it shows that “AI” is not one algorithm. It is a full pipeline.

### 4. AI methods fit into the four ingredients

Percy previews the course by attaching methods to the ingredients:

- perception: vision, speech, language understanding
- reasoning: uniform cost search, value iteration, minimax, probabilistic inference
- action: generation, speech synthesis, robot control
- learning: gradient descent, Q-learning, expectation maximization

A subtle but important point: the course is heavy on reasoning methods. That makes CS221 broader than a purely deep-learning course.

### 5. Resource constraints

This is one of the most important conceptual slides of the lecture.

All intelligent behavior must operate under limited resources:

#### Computation constraints
- limited time
- limited memory
- limited communication

#### Information constraints
- incomplete observations
- finite data
- finite experience
- local, not omniscient, access to the world

This means AI is often about approximation, prioritization, and structure. You need smart algorithms precisely because brute force is impossible.

#### Analogy

Imagine trying to play chess by evaluating every future possibility to the end of the game. In theory, that sounds ideal. In practice, there is not enough time. Intelligence often means finding the right shortcut without losing too much quality.

### 6. Goals: developer goals vs societal goals

Percy then deepens the question from “what can an AI do?” to “what should an AI do?”

There are two levels:

#### Developer goals
The builder wants the agent to optimize something. That objective may be stated directly or hidden in training procedures and design choices.

Example for a chatbot:
- be informative
- avoid hallucinations
- refuse harmful requests

This is the alignment problem at the developer level.

#### Societal goals
Even if a product does what its creator intended, society may still face harms involving:
- privacy
- copyright
- jobs
- inequality
- geopolitics
- unintended downstream consequences

This is why Percy calls AI a **sociotechnical** problem. The technical system and the social system interact.

### 7. What this course emphasizes

The course is about principles and techniques.

Main philosophy:
- timeless foundations
- modern examples
- learning by doing

He emphasizes coding and building systems, not just reading theory. That is consistent with AI as an empirical engineering discipline.

### 8. Why the course is tensor-native

This is one of the most important practical announcements in the lecture.

The course has moved to a “tensor-native” style built around NumPy and PyTorch.

Why?

Because tensors are not merely a deep learning trick. They are a general computational substrate for many AI algorithms.

This is a big idea:

- if you can express a computation as tensor operations
- then you often gain clarity, efficiency, GPU compatibility, and scalability

So tensor notation is being taught early because it becomes the common language for later topics.

### 9. Executable lectures

Percy explains that the lecture itself is generated from executable code.

Why this matters:
- code gives hierarchical structure
- code is precise
- code has execution semantics
- AI systems are ultimately implemented in code anyway

This is a pedagogical choice. The lecture is not merely describing algorithms; it is already living in a computational medium.

---

### 10. History of AI: the Turing Test

Percy begins the history section with Alan Turing’s 1950 paper and the imitation game.

The key significance is not that the Turing Test is a perfect definition of intelligence. It is that Turing turned a philosophical question into an operational one.

That is a recurring AI pattern:

- define a measurable task
- evaluate systems objectively
- iterate based on benchmarks

This “benchmark culture” still shapes AI today.

### 11. Symbolic AI

Symbolic AI emphasizes:
- logic
- search
- knowledge representation
- explicit structure

Important examples in the lecture:
- Dartmouth workshop (1956)
- Arthur Samuel’s checkers program
- Newell and Simon’s Logic Theorist
- expert systems such as DENDRAL, MYCIN, XCON

#### Why symbolic AI was powerful
It had a clear vision of intelligent reasoning:
- represent knowledge explicitly
- manipulate symbols logically
- search for solutions systematically

#### Why it struggled
Two big obstacles:
- search spaces exploded combinatorially
- hand-written rules became brittle and unmanageable

#### Key lesson
Explicit knowledge is valuable, but the world is messy, uncertain, and large. Pure rule-based approaches often fail under that complexity.

### 12. Neural AI

Neural AI emphasizes:
- learnable representations
- differentiable computation
- gradient-based optimization
- layered function approximation

Important milestones in the lecture:
- McCulloch and Pitts
- Hebbian learning
- perceptron
- ADALINE
- criticism from Minsky/Papert
- backpropagation revival
- CNNs
- AlexNet
- seq2seq
- attention
- AlphaGo
- transformers

#### Why neural AI became dominant
Because it scales well with:
- data
- compute
- optimization improvements
- hardware improvements

#### Key lesson
Neural methods became extremely strong first in perception, then increasingly in language and reasoning-style tasks.

### 13. Statistical AI

Statistical AI emphasizes:
- probability
- uncertainty
- inference
- optimization
- generalization

Important ingredients mentioned:
- linear regression
- stochastic gradient descent
- uniform cost search
- Markov decision processes
- Bayesian networks
- support vector machines
- variational inference
- conditional random fields
- topic models

#### Why this tradition matters
It gave AI much of its mathematical rigor:
- how to think about uncertainty
- how to optimize objectives
- how to reason about training vs generalization

Percy’s point is that modern AI still depends heavily on this tradition, even when the models themselves are neural.

### 14. Foundation models and industrialization

The lecture then moves into the recent era:
- pretrained language models
- scaling laws
- GPT-style large language models
- reasoning models
- enormous training clusters
- decreasing openness around frontier models

This section is not mainly historical trivia. It supports two deeper points:

1. AI has become industrialized at enormous scale.
2. Despite huge progress, many deep questions about intelligence remain unsolved.

### 15. Synthesis across traditions

One of the best conceptual conclusions in the lecture is that the traditions are not mutually exclusive.

Percy’s synthesis:
- symbolic AI provided the vision and ambition
- neural AI provided model architectures
- statistical AI provided rigor

This is a very good way to organize your understanding of modern AI.

---

### 16. Tensors: the atoms of modern ML

Now the lecture turns practical.

A tensor is a multi-dimensional array.

Examples:

- scalar: one number, shape `()`
- vector: one axis, shape like `(3,)`
- matrix: two axes, shape like `(2, 3)`
- rank-3 tensor: shape like `(2, 2, 3)`

#### Practical mental model

A tensor is just a structured container of numbers with axes.

Do not mystify the word.

- scalar = single number
- vector = line of numbers
- matrix = table of numbers
- rank-3 tensor = stack of tables
- rank-4 tensor = stack of stacks of tables

That is the right beginner mental model.

#### Analogy

Think of a tensor like an Excel workbook:

- scalar = one cell
- vector = one row or one column
- matrix = one sheet
- rank-3 tensor = several sheets stacked together
- rank-4 tensor = several workbooks of sheets

The “shape” tells you how many entries exist along each organizational direction.

### 17. Creating tensors

Percy shows how tensors are created in NumPy using:
- direct construction
- zeros
- ones
- random normal values
- identity matrices
- diagonal matrices
- loading/saving from disk

The important conceptual point is not the syntax itself. It is that tensors are the standard container type for computation.

### 18. Shapes and indexing

Given a tensor, you can index into it progressively.

For a tensor with shape `(2, 2, 3)`:
- indexing once selects one of the 2 outer blocks
- indexing again selects one row within that block
- indexing again selects one entry within that row

This is just nested structure.

#### Key study habit

Whenever you slice a tensor, ask:
- what axis am I selecting over?
- what shape remains afterward?

If you can answer that consistently, tensor indexing becomes mechanical rather than mysterious.

### 19. Typical tensors in machine learning

This part is very important because it connects abstract tensor notation to actual ML objects.

#### A single feature vector
A single data point with $D$ features:
$$
x \in \mathbb{R}^{D}
$$

#### A dataset of feature vectors
$N$ examples, each with $D$ features:
$$
X \in \mathbb{R}^{N \times D}
$$

Here:
- axis 0 = which example
- axis 1 = which feature

#### A batch of token sequences
If each example is a sequence of length $L$, and each token position has a $D$-dimensional representation:
$$
X \in \mathbb{R}^{N \times L \times D}
$$

Here:
- axis 0 = which sequence in the batch
- axis 1 = which token position
- axis 2 = which hidden feature

#### A batch of images
If images have height $H$, width $W$, and channels $C$:
$$
X \in \mathbb{R}^{N \times H \times W \times C}
$$

Here:
- axis 0 = which image
- axis 1 = vertical position
- axis 2 = horizontal position
- axis 3 = color channel

#### A weight matrix
To map $D_{\text{in}}$ inputs to $D_{\text{out}}$ outputs:
$$
W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}
$$

This is how a linear layer is stored.

### 20. Why batching matters

Batching means grouping multiple examples together into one tensor so the same operation can be applied to all of them at once.

Why do this?

- it is computationally efficient
- it uses optimized linear algebra kernels
- it maps naturally to GPU hardware
- it makes code cleaner

#### Analogy

Instead of processing exam papers one at a time, imagine feeding a whole stack through a scanner at once. The same operation is applied repeatedly, but the machine works much more efficiently in batch mode.

### 21. Views, slicing, and mutation

Percy emphasizes a subtle but very important point: many operations such as slicing or transpose create a **view**, not a fresh independent copy.

That means two different tensor objects can reference the same underlying memory.

So if you mutate one, the other may appear to change too.

#### Why this matters

This is a classic source of bugs:
- you think you created a harmless transformed tensor
- you modify it
- your original tensor unexpectedly changes

#### Study takeaway
At this stage, prefer a mostly functional style:
- create new tensors
- avoid unnecessary mutation
- treat shape transformations carefully

### 22. Elementwise operations

Elementwise operations act independently on each entry:
- square
- square root
- add
- multiply by scalar
- divide by scalar

These preserve shape.

Also introduced:
- `triu`: upper triangular part
- `tril`: lower triangular part

These are useful for masking, especially in transformers where future positions may need to be hidden from earlier positions.

### 23. Matrix multiplication: the main event

Percy calls matrix multiplication the bread and butter of deep learning.

If
$$
X \in \mathbb{R}^{4 \times 6}, \quad W \in \mathbb{R}^{6 \times 3},
$$
then
$$
Y = XW \in \mathbb{R}^{4 \times 3}.
$$

Inner dimensions must match:
$$
(4 \times 6)(6 \times 3) \to (4 \times 3)
$$

#### Interpretation
Each row of the result is formed by taking one row of $X$ and multiplying it by $W$.

You can think of this as applying the same linear transformation to multiple input vectors at once.

### 24. Batched matrix multiplication

Now suppose
$$
X \in \mathbb{R}^{2 \times 4 \times 6}, \quad W \in \mathbb{R}^{6 \times 3}.
$$

Then
$$
Y = XW \in \mathbb{R}^{2 \times 4 \times 3}.
$$

What happened?

The same matrix $W$ is applied to each slice `X[0]`, `X[1]`, and so on.

This is one of the most important tensor patterns in ML.

#### Beginner interpretation

Do not think:
- “magic tensor wizardry”

Think:
- “I had a pile of matrices, and I multiplied each of them by the same matrix.”

That is all.

### 25. Efficiency and vectorization

Percy compares:
- a manual Python triple-loop matrix multiplication
- a NumPy matrix multiplication

The NumPy version is much faster.

#### Why?
Because NumPy / PyTorch:
- use optimized low-level implementations
- exploit memory layout carefully
- can leverage SIMD and parallelism
- can run on GPUs
- integrate well with distributed computation

This is why tensor programming matters so much. It is not only mathematically elegant; it is also close to the hardware and systems reality of modern AI.

### 26. Why tensor code can feel hard

A major pain point Percy acknowledges is that tensor code can become less readable, especially when high-rank tensors are involved.

This is exactly where many students get lost.

The difficulty is usually not the arithmetic itself. The difficulty is bookkeeping:
- which axis means what?
- which dimensions are being multiplied?
- which ones are being summed out?
- which ones stay?

That is why named-dimension thinking is so valuable.

### 27. `einops` / Einstein-style notation

Percy introduces `einops` as a cleaner way to write tensor computations.

Core idea:
- name dimensions by meaning rather than by position

Example pattern:
- `batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2`

This says:
- there are two input tensors
- they share `batch`
- they each have a sequence dimension and a hidden dimension
- hidden is summed over
- the output keeps `batch`, `seq1`, and `seq2`

This is far easier to reason about than remembering that you transposed axes `-2` and `-1`.

### 28. `einsum`: generalized matrix multiplication

`einsum` is best understood as:

- label axes
- state which axes survive
- any axis not on the output gets summed over

That is the whole trick.

#### Mathematical interpretation

If you write something like:
$$
\text{batch seq1 hidden, batch seq2 hidden} \to \text{batch seq1 seq2},
$$
then `hidden` disappears from the output, so it is summed over.

This mirrors ordinary matrix multiplication:
$$
Y_{ij} = \sum_k X_{ik} W_{kj}
$$

Here the shared inner index is what gets summed out.

### 29. `reduce`

`reduce` takes one tensor and collapses one or more axes using an operation such as:
- sum
- mean
- max
- min

Again, the easiest way to reason is:
- any axis not named in the output is being reduced away

### 30. `rearrange`

`rearrange` changes how dimensions are organized.

This is especially useful when one dimension really represents multiple logical dimensions packed together.

Example idea:
- a hidden dimension of size 8 might secretly mean `heads = 2` and `hidden_per_head = 4`

Then `rearrange` can:
- unflatten `(heads hidden)`
- perform an operation per head
- flatten again

This is extremely common in transformer implementations.

#### Intuition

Imagine a suitcase with 8 items inside. If you later realize those 8 items are really 2 folders of 4 items each, `rearrange` is the act of unpacking the suitcase into folders, working at the folder level, then repacking it.

---

## Summary

Lecture 1 does three jobs.

First, it defines AI operationally through four ingredients:
- perception
- reasoning
- action
- learning

Second, it situates the field historically as an interaction between:
- symbolic AI
- neural AI
- statistical AI

Third, it introduces tensors as the computational language that will unify much of the course.

The single most important practical takeaway is this:

**If you can track tensor shapes confidently, a huge amount of modern AI becomes much easier to understand.**

The single most important conceptual takeaway is this:

**AI is not just about making systems powerful. It is about making systems act intelligently under limited computation, limited information, and competing human values.**

---

## Real-World Applications

### 1. Large language models
LLMs represent:
- token embeddings as tensors
- attention scores as tensors
- parameter matrices as tensors
- gradients as tensors

Almost every forward and backward pass is tensor algebra.

### 2. Computer vision
Images naturally form tensors:
$$
N \times H \times W \times C
$$
Convolutions, feature maps, pooling, and attention are all tensor operations.

### 3. Reinforcement learning
State values, Q-values, transition models, and policy outputs can all be represented and updated using tensor operations.

This is why Percy says even methods like value iteration fit naturally into tensor-native computation.

### 4. Probabilistic inference
Probabilities over variables, factors, and marginalizations can be represented as tensors and manipulated through reductions and structured multiplications.

This is why tensor thinking also matters for Bayesian networks.

### 5. Scientific computing
Outside AI, tensors appear in:
- physics
- chemistry
- signal processing
- numerical simulation
- computational biology

So this lecture is not only preparing you for CS221. It is giving you a broadly useful computational language.

---

## My Explanations for the Most Likely Confusions

### Confusion 1: “A tensor sounds like some scary advanced object.”

For this course, start with the simplest possible view:

A tensor is just a multi-dimensional array of numbers.

That is enough for now.

### Confusion 2: “Why not just use loops?”

Because loops in Python are slow, harder to parallelize, and further from the optimized linear algebra stack that modern AI systems rely on.

Tensor operations let you express the computation at the right level of abstraction.

### Confusion 3: “I can do the arithmetic, but I lose track of the dimensions.”

That is normal. The fix is to annotate every important tensor with dimension names.

For example:
- `batch`
- `seq`
- `hidden`
- `heads`
- `height`
- `width`
- `channels`

Once dimensions have semantic names, the operations become far easier to follow.

### Confusion 4: “Broadcasting feels magical.”

It is less magical if you read it as:

“A smaller tensor is being reused across extra axes of a larger tensor.”

Same operation, repeated systematically.

### Confusion 5: “Why does Percy care so much about notation?”

Because notation is not decoration. In AI, notation determines whether high-dimensional computation is understandable, implementable, and debuggable.

Good tensor notation is like good accounting notation or good programming style: it prevents expensive mistakes.

---

## Tensor Cheat Sheet

### Canonical shapes

- scalar: `()`
- vector: `(D,)`
- matrix / batch of feature vectors: `(N, D)`
- batch of sequences: `(N, L, D)`
- batch of images: `(N, H, W, C)`
- weight matrix: `(D_in, D_out)`

### Rules of thumb

- always write down what each axis means
- the shape tells you the semantics of the computation
- in matrix multiplication, inner dimensions must match
- axes omitted from an `einsum` output are summed over
- slicing and transpose may create views rather than copies
- batching usually sits on the leftmost axis
- if a tensor expression looks confusing, rewrite it with named dimensions

---

## Final Takeaway

Lecture 1 is not “just an intro.”

It quietly sets up the whole course.

- The philosophy of intelligence: perceive, reason, act, learn
- The constraints: compute and information are limited
- The societal lens: goals and impacts matter
- The historical lens: modern AI comes from multiple traditions
- The computational lens: tensors are the shared language for implementing AI systems

If you master the tensor material early, the rest of the course will feel much more coherent.

In practice, that means your near-term goal should be:

**become fluent at reading shapes, predicting output shapes, and interpreting tensor operations in words before trying to memorize syntax.**
