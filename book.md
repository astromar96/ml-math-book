# Math Foundations for Machine Learning — Made Easy

## How to use this book

You need only three areas of math for machine learning: linear algebra, calculus, and probability & statistics. This book teaches each one with everyday intuition first, then the formula, then a small worked example, then the link to ML.

**Who it's for:** anyone heading toward a junior ML engineer role who finished high-school math (or forgot most of it). No proofs, no heavy theory.

**How each chapter works:**

1. **Intuition** — what the idea means in plain words.
2. **The math** — the formula, kept as short as possible.
3. **Worked example** — real numbers, every step shown.
4. **Why ML cares** — where you will meet it in real models.
5. **Practice** — a few problems, with answers at the end of the book.

**Study tips:**

- Study 30–60 minutes a day. Short and steady beats long and rare.
- Work every example with pen and paper before reading the solution.
- Check your answers in Python with NumPy. Each chapter shows the one-line version.
- Stuck for more than 20 minutes? Move on and come back tomorrow.
- Don't aim for perfect. Aim for "I can explain this to a friend."

> **بالعربي:** إنت محتاج تلات مجالات بس: الجبر الخطي، التفاضل، والاحتمالات والإحصاء. وكل موضوع في الكتاب ماشي بنفس الترتيب: الفكرة بالكلام العادي ← الصيغة ← مثال بأرقام ← ليه ML محتاجها ← تمارين وحلولها في آخر الكتاب. ونصايح سريعة: ذاكر 30–60 دقيقة في اليوم، حل بالقلم والورقة قبل ما تبصّ على الحل، اتأكد من إجابتك بـ NumPy، ولو زنقت أكتر من 20 دقيقة سيبها وارجعلها بكره. الهدف مش الكمال — الهدف إنك تقدر تشرح الفكرة لصاحبك.

## Chapter 1: Math warm-up

This chapter refreshes the five small tools every later chapter uses: notation, functions, exponents, logarithms, and summation.

> **بالعربي:** الشابتر ده بيسخّن معاك على الخمس أدوات الصغيرة اللي كل الشابترات اللي بعده مبنية عليها: الرموز، الدوال، الأسس، اللوغاريتمات، والمجموع (Σ). لو فاكرهم من المدرسة خده كمراجعة سريعة.

### 1.1 Reading math notation

Math symbols are just shorthand. Learn them once and papers stop looking scary.

| Symbol | Read it as | Example |
| --- | --- | --- |
| x, y | a number (a scalar) | x = 3 |
| **x** (bold) | a vector (a list of numbers) | **x** = \[2, 5, 1\] |
| **X** (bold capital) | a matrix (a grid of numbers) | a 3 × 2 table |
| Σ | "add them all up" | Σ xᵢ |
| ∂ | "partial derivative" | ∂f/∂x |
| ∇ | "gradient" | ∇f |
| ∈ | "belongs to" | x ∈ ℝ (x is a real number) |
| ŷ | "y-hat", a prediction | ŷ = 4.2 |
| P(A) | probability of A | P(rain) = 0.3 |

> **بالعربي:** الرموز دي مش أكتر من اختصارات، اتعلمها مرة واحدة وأي paper بعد كده هتقراه عادي:
>
> - **x, y** رقم عادي (scalar).
> - **x** بالبولد يعني ليستة أرقام (vector)، و **X** بالكابيتال يعني جدول أرقام (matrix).
> - **Σ** معناها "اجمع كله"، و **∂** مشتقة جزئية (غيّر متغير واحد بس)، و **∇** الـ gradient (المشتقات الجزئية كلها مع بعض).
> - **∈** يعني "ينتمي لـ"، و **ŷ** معناها توقّع الموديل، و **P(A)** احتمال حصول A.

### 1.2 Functions

**Intuition:** a function is a machine. You put a number in, you get a number out.

```latex
f(x) = 2x + 1
```

Put in x = 3 and you get f(3) = 2 × 3 + 1 = 7.

**Why ML cares:** a trained model *is* a function. It takes features (house size, rooms) and returns a prediction (price).

> **بالعربي:** الدالة (function) زي الماكينة: تدخّلها رقم، تطلّعلك رقم. f(x) = 2x + 1 معناها "اضرب في 2 وزوّد 1"، فلو دخّلت 3 يطلعلك 7. وأي موديل مدرَّب هو في الآخر دالة: بتدخّله بيانات البيت (المساحة، عدد الأوض) ويطلّعلك السعر.

### 1.3 Exponents

**Intuition:** an exponent says how many times to multiply a number by itself.

- 2³ = 2 × 2 × 2 = 8
- x⁰ = 1 for any non-zero x
- x⁻¹ = 1/x
- xᵃ × xᵇ = xᵃ⁺ᵇ

> **بالعربي:** الأس بيقولك تضرب الرقم في نفسه كام مرة:
>
> - 2³ = 2 × 2 × 2 = 8.
> - أي رقم (مش صفر) أس صفر = 1.
> - الأس السالب معناه مقلوب: x⁻¹ = 1/x.
> - ولو بتضرب نفس الأساس، اجمع الأسس: xᵃ × xᵇ = xᵃ⁺ᵇ.


The special number **e ≈ 2.718** shows up everywhere in ML. The sigmoid function, used in logistic regression, is built from it:

```latex
\sigma(x) = \frac{1}{1 + e^{-x}}
```

It squashes any number into the range 0 to 1, so it can act like a probability.

> **بالعربي:** دالة السيجمويد بتاخد أي رقم (سواء كبير أوي أو سالب أوي) وبتحوّله لرقم ما بين 0 و 1. تخيّلها زي مقياس بيقولك "أد إيه أنا متأكد" — 0 يعني مش متأكد خالص، و 1 يعني متأكد جدًا. عشان كده بنستخدمها في التصنيف.

### 1.4 Logarithms

**Intuition:** a log is the reverse of an exponent. It asks: "what power do I need?"

- log₂(8) = 3, because 2³ = 8
- log₁₀(100) = 2, because 10² = 100
- ln(x) means log base e

Useful rules:

- log(a × b) = log(a) + log(b) — turns multiplication into addition
- log(1) = 0
- log of a number between 0 and 1 is negative

> **بالعربي:** اللوغاريتم هو عكس الأس، بيسأل: "محتاج أرفع الأساس لأنهي قوة؟" يعني log₂(8) = 3 لأن 2³ = 8، و ln معناها لوغاريتم للأساس e. والقواعد المفيدة:
>
> - log(a × b) = log(a) + log(b) — بيحوّل الضرب لجمع.
> - log(1) = 0 دايمًا.
> - لوغاريتم أي رقم ما بين 0 و 1 بيطلع سالب.


**Why ML cares:** multiplying many small probabilities gives tiny numbers computers struggle with. Taking logs turns them into sums. This is why the classification loss is called *log loss* (cross-entropy).

> **بالعربي:** لما بنضرب احتمالات صغيرة في بعض، النتيجة بتبقى رقم صغير جدًا الكمبيوتر مش بيعرف يتعامل معاه. اللوغاريتم بيحوّل الضرب لجمع، والأرقام الصغيرة لأرقام سالبة عادية. ده السبب إن اسم دالة الخطأ في التصنيف "log loss".

### 1.5 Summation (Σ)

**Intuition:** Σ is a for-loop written in math.

```latex
\sum_{i=1}^{4} x_i = x_1 + x_2 + x_3 + x_4
```

If **x** = \[3, 1, 4, 2\], the sum is 3 + 1 + 4 + 2 = 10.

In Python: `sum(x)` or `np.sum(x)`.

> **بالعربي:** الـ Σ مش حاجة مخيفة — هي for-loop مكتوبة بالرياضة. Σ من i=1 لـ 4 معناها "لفّ على العناصر من الأول للرابع واجمعهم". ولو **x** = \[3, 1, 4, 2\] يبقى المجموع 10. في بايثون سطر واحد: `sum(x)`.

### Practice — Chapter 1

1. If f(x) = 3x − 2, what is f(4)?
2. Simplify 2⁴ × 2².
3. What is log₂(32)?
4. Compute Σ i² for i = 1 to 3.
5. What is σ(0) for the sigmoid function?

## Chapter 2: Linear algebra

Linear algebra is the language of data: every dataset is a matrix, every data point is a vector, and almost every model is matrix multiplication.

> **بالعربي:** الجبر الخطي هو لغة البيانات: أي داتاسِت هو مصفوفة، وأي نقطة بيانات هي vector، وأغلب الموديلات في الآخر مجرد ضرب مصفوفات. لو فهمت الشابتر ده، فهمت نص الـ ML.

### 2.1 Vectors

**Intuition:** a vector is a list of numbers that describes one thing. A house might be **x** = \[120, 3, 2\], meaning 120 m², 3 rooms, 2 bathrooms.

Geometrically, a vector is an arrow from the origin. \[3, 4\] points 3 steps right and 4 steps up.

**Adding vectors:** add position by position.

- \[1, 2\] + \[3, 5\] = \[4, 7\]

**Scaling a vector:** multiply every entry.

- 3 × \[1, 2\] = \[3, 6\]

**Length (norm):** how long the arrow is, by Pythagoras.

```latex
\|\mathbf{x}\| = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}
```

Example: the length of \[3, 4\] is √(9 + 16) = √25 = 5.

> **بالعربي:**
>
> - **الـ vector** ليستة أرقام بتوصف حاجة واحدة: البيت = \[120, 3, 2\] يعني 120 متر، 3 أوض، 2 حمام. وهندسيًا هو سهم بيبدأ من نقطة الأصل.
> - **الجمع:** بتجمع كل خانة مع الخانة اللي قصادها بالظبط، مش كله في كله.
> - **الضرب في رقم (scaling):** بتضرب كل خانة في الرقم — السهم يطول أو يقصر وهو ثابت في نفس الاتجاه.
> - **الطول (norm):** فيثاغورس بس في أبعاد أكتر: ربّع كل خانة، اجمع، خُد الجذر. طول \[3, 4\] = 5.

### 2.2 The dot product

**Intuition:** the dot product measures how much two vectors point the same way. Multiply matching entries, then add.

```latex
\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n
```

Worked example: \[1, 2, 3\] · \[4, 5, 6\] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = **32**.

What the sign tells you:

- Positive → the vectors point roughly the same way.
- Zero → they are at a right angle (unrelated).
- Negative → they point roughly opposite ways.

**Why ML cares:** a linear model's prediction is one dot product: weights · features + bias. Recommendation systems and search rank items by dot products of embeddings (cosine similarity).

> **بالعربي:** حاصل الضرب النقطي (dot product) ببساطة بيقيس "أد إيه الحاجتين دول متشابهين في الاتجاه". لو النتيجة كبيرة وموجبة يبقى متشابهين، لو صفر يبقى مالهمش علاقة ببعض، ولو سالبة يبقى عكس بعض. الموديل بيتعلم عن طريق ضرب الأوزان (weights) في البيانات (features) — وده حرفيًا dot product واحد.

### 2.3 Matrices

**Intuition:** a matrix is a table of numbers. In ML, each row is one example and each column is one feature.

|  | Size (m²) | Rooms | Age (years) |
| --- | --- | --- | --- |
| House 1 | 120 | 3 | 10 |
| House 2 | 85 | 2 | 25 |
| House 3 | 200 | 5 | 2 |

This is a 3 × 3 matrix: 3 rows × 3 columns. The shape is always written rows × columns.

> **بالعربي:** الـ matrix جدول أرقام. وفي ML، كل **صف** مثال واحد (بيت، عميل، صورة) وكل **عمود** خصيصة واحدة (feature). والشكل (shape) بيتكتب دايمًا صفوف × أعمدة، والترتيب ده هو اللي بيحدّد الضرب هيمشي ولا لأ.

### 2.4 Matrix multiplication

**The one rule to remember:** each entry of the result is a dot product of a **row** from the first matrix with a **column** from the second.

Shapes must line up: (m × **n**) times (**n** × p) gives (m × p). The inner numbers must match.

Worked example:

```latex
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
=
\begin{bmatrix} 1\cdot5 + 2\cdot7 & 1\cdot6 + 2\cdot8 \\ 3\cdot5 + 4\cdot7 & 3\cdot6 + 4\cdot8 \end{bmatrix}
=
\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

Watch out: order matters. A × B is usually **not** equal to B × A.

> **بالعربي:** ضرب المصفوفات مش زي ضرب الأرقام العادية. A × B مش هيطلع زي B × A. القاعدة: كل خانة في النتيجة هي dot product بين صف من المصفوفة الأولى وعمود من التانية. ولازم الرقم الجوّاني يتطابق: (صفوف × **أعمدة**) ضرب (**صفوف** × أعمدة) — لو مش متطابقين، الضرب مش ممكن.

**Why ML cares:** predicting for a whole dataset at once is one multiplication, **X** × **w**. A neural network layer is a matrix multiplication followed by an activation function. GPUs exist to do this fast.

In Python: `A @ B` or `np.dot(A, B)`.

### 2.5 Transpose

**Intuition:** flip the matrix so rows become columns.

```latex
\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
```

A 2 × 3 matrix becomes 3 × 2. In Python: `A.T`.

**Why ML cares:** you transpose constantly to make shapes line up for multiplication.

> **بالعربي:** الـ transpose يعني تقلب المصفوفة: الصفوف تبقى أعمدة والأعمدة تبقى صفوف، فـ 2 × 3 تبقى 3 × 2. هتعمله كل شوية عشان تخلّي الأشكال (shapes) تتطابق قبل الضرب. في بايثون سطر واحد: `A.T`.

### 2.6 Identity and inverse

**Identity matrix (I):** 1s on the diagonal, 0s elsewhere. It is the matrix version of the number 1: A × I = A.

**Inverse (A⁻¹):** the matrix version of division. A × A⁻¹ = I. Not every matrix has one, just as you can't divide by 0.

For a 2 × 2 matrix:

```latex
\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
```

The number ad − bc is the **determinant**. If it is 0, there is no inverse.

**Why ML cares:** linear regression has an exact solution (the normal equation) that uses an inverse. In practice, libraries avoid computing inverses directly because it is slow and unstable.

> **بالعربي:**
>
> - **مصفوفة الوحدة (I):** واحدات على القطر وأصفار في الباقي. هي زي الرقم 1 بالظبط: A × I = A.
> - **المعكوس (A⁻¹):** نسخة القسمة في عالم المصفوفات: A × A⁻¹ = I. ومش كل مصفوفة ليها معكوس، زي ما إنت مش بتقدر تقسم على صفر.
> - **المحدِّد (determinant) = ad − bc:** لو طلع صفر يبقى مفيش معكوس خالص.
>
> ملاحظة عملية: المكتبات بتتجنّب حساب المعكوس على المباشر لأنه بطيء وغير مستقر، وبتستخدم طرق تانية بتوصل لنفس النتيجة.

### 2.7 Eigenvectors and eigenvalues (intuition only)

**Intuition:** most vectors change direction when multiplied by a matrix. An **eigenvector** is a special direction that only gets stretched or shrunk, never turned. The **eigenvalue** is how much it gets stretched.

```latex
A\mathbf{v} = \lambda \mathbf{v}
```

Example: for A = \[\[2, 0\], \[0, 3\]\], the vector \[1, 0\] becomes \[2, 0\]. Same direction, twice as long. So \[1, 0\] is an eigenvector with eigenvalue 2.

**Why ML cares:** PCA (dimensionality reduction) finds the eigenvectors of the data's covariance matrix. These are the directions where the data spreads out most, so you can keep the important ones and drop the rest.

> **بالعربي:** تخيّل إن عندك سحابة نقاط في الفضاء. الـ eigenvector هو الاتجاه اللي النقاط ممتدة فيه أكتر حاجة — زي محور السحابة الرئيسي. الـ eigenvalue بيقولك قد إيه النقاط ممتدة في الاتجاه ده. الـ PCA بيستخدم ده عشان يلاقي أهم الاتجاهات ويشيل الباقي، فتقلل حجم البيانات من غير ما تخسر المعلومات المهمة.

### 2.8 Linear algebra in one ML line

```latex
\hat{\mathbf{y}} = X\mathbf{w} + b
```

- X — the data matrix (examples × features)
- **w** — the weights the model learns
- b — the bias
- ŷ — predictions for every example at once

That single line is linear regression, and the first step of every neural network layer.

> **بالعربي:** السطر ŷ = Xw + b هو الجبر الخطي كله في جملة واحدة: X البيانات (أمثلة × فيتشرز)، و w الأوزان اللي الموديل بيتعلّمها، و b الإزاحة (bias)، و ŷ التوقّعات لكل الأمثلة مرة واحدة. السطر ده بعينه هو الـ linear regression، وهو كمان أول خطوة في أي طبقة من أي شبكة عصبية.

### Practice — Chapter 2

1. Compute \[2, −1, 3\] + \[1, 4, 0\].
2. Find the length of \[6, 8\].
3. Compute the dot product \[1, 0, 2\] · \[3, 5, 4\].
4. Are \[1, 2\] and \[−2, 1\] at a right angle? Why?
5. A is 4 × 3 and B is 3 × 2. What is the shape of A × B? Can you compute B × A?
6. Multiply \[\[1, 0\], \[2, 1\]\] by \[\[3, 1\], \[0, 2\]\].
7. Find the determinant of \[\[4, 2\], \[3, 1\]\]. Does it have an inverse?

## Chapter 3: Calculus

In ML, calculus does one job: it tells the model which way to adjust its weights so the error goes down. That's all you need from it.

> **بالعربي:** التفاضل في ML بيعمل حاجة واحدة بس: بيقول للموديل يظبّط أوزانه في أنهي اتجاه عشان الخطأ يقل. ومش محتاج منه أكتر من كده خالص.

### 3.1 The derivative

**Intuition:** the derivative is the slope, or how fast something changes. If you drive and your position changes 60 km every hour, the derivative of position is 60 km/h.

On a graph, the derivative at a point is the steepness of the curve right there:

- Positive → the curve goes up as x grows.
- Negative → the curve goes down.
- Zero → a flat spot: a bottom, a top, or a plateau.

```latex
f'(x) = \frac{df}{dx} \approx \frac{f(x + h) - f(x)}{h} \text{ for a tiny } h
```

Worked example: for f(x) = x², try x = 3 and h = 0.001. Then (3.001² − 3²) / 0.001 = (9.006001 − 9) / 0.001 ≈ **6**. The exact answer is 2 × 3 = 6.

> **بالعربي:** المشتقة (derivative) هي الميل: "الحاجة بتتغير بأي سرعة؟" لو بتسوق وموقعك بيتغير 60 كم كل ساعة، يبقى مشتقة الموقع 60 كم/س. وعلى الرسم:
>
> - **موجبة** → المنحنى طالع مع زيادة x.
> - **سالبة** → المنحنى نازل.
> - **صفر** → نقطة مستوية: قمة أو قاع أو سطح مستوي.
>
> وده بالظبط اللي الموديل محتاجه: يعرف الخطأ طالع ولا نازل عشان يعرف يمشي في أنهي اتجاه.

### 3.2 The rules you actually need

You don't compute slopes by hand. You use these rules:

| Function | Derivative | Example |
| --- | --- | --- |
| constant c | 0 | (5)' = 0 |
| xⁿ (power rule) | n·xⁿ⁻¹ | (x³)' = 3x² |
| c·f(x) | c·f'(x) | (4x²)' = 8x |
| f(x) + g(x) | f'(x) + g'(x) | (x² + 3x)' = 2x + 3 |
| eˣ | eˣ | — |
| ln(x) | 1/x | — |
| f(x)·g(x) (product rule) | f'g + fg' | (x·eˣ)' = eˣ + x·eˣ |

Worked example: f(x) = 3x⁴ − 2x² + 7x − 5.

- 3x⁴ → 12x³
- −2x² → −4x
- 7x → 7
- −5 → 0

So f'(x) = 12x³ − 4x + 7.

> **بالعربي:** مش هتحسب الميل بإيدك، هتستعمل القواعد دي وخلاص:
>
> - **الثابت** مشتقته صفر — الرقم الثابت مابيتغيرش.
> - **قاعدة القوة (power rule):** نزّل الأس قدّام واطرح منه واحد: (x³)' = 3x².
> - **رقم × دالة:** الرقم بيقعد زي ما هو: (4x²)' = 8x.
> - **جمع دوال:** اشتق كل واحدة لوحدها واجمع.
> - **eˣ** مشتقته نفسه (الحالة الوحيدة اللي بتعمل كده)، و **ln(x)** مشتقته 1/x.
> - **قاعدة الضرب (product rule):** f'g + fg'.

### 3.3 Partial derivatives

**Intuition:** a model has many weights, so the error depends on many variables. A partial derivative asks: "if I nudge only *this one* weight and freeze the rest, how does the output change?"

Rule: treat every other variable as a constant, then differentiate as usual.

Worked example: f(x, y) = x²y + 3y.

- ∂f/∂x = 2xy (y is treated as a constant)
- ∂f/∂y = x² + 3 (x is treated as a constant)

> **بالعربي:** المشتقة الجزئية (partial derivative) ببساطة بتقولك: "لو غيّرت متغير واحد بس وثبّت كل حاجة تانية، الناتج هيتغير أد إيه؟" في الموديل، عندك أوزان كتير. المشتقة الجزئية بتقولك كل وزن لوحده مأثّر على الخطأ أد إيه. ودي اللي بنسمّيه الـ gradient — مجموعة المشتقات الجزئية كلها مع بعض.

### 3.4 The gradient

**Intuition:** the gradient collects all the partial derivatives into one vector. It points in the direction of steepest **uphill**.

```latex
\nabla f = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right]
```

From the example above, at the point (1, 2): ∇f = \[2·1·2, 1² + 3\] = \[4, 4\].

To go **downhill** (reduce error), walk in the opposite direction: −∇f.

> **بالعربي:** الـ gradient (∇f) هو المشتقات الجزئية كلها محطوطة في vector واحد، وهو بيشاور على اتجاه **الطلوع** الأسرع. وإحنا عايزين ننزل (نقلّل الخطأ)، فبنمشي في الاتجاه المعاكس تمامًا: −∇f. الجملة دي هي كل حكاية "تدريب الموديل".

### 3.5 The chain rule

**Intuition:** when functions are stacked inside each other, multiply their slopes. If y changes 3 times as fast as u, and u changes 2 times as fast as x, then y changes 3 × 2 = 6 times as fast as x.

```latex
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
```

Worked example: y = (3x + 1)².

1. Let u = 3x + 1, so y = u².
2. dy/du = 2u and du/dx = 3.
3. dy/dx = 2u × 3 = 6(3x + 1).

**Why ML cares:** a neural network is many functions stacked in layers. **Backpropagation** is just the chain rule applied layer by layer, from the output back to the input. PyTorch and TensorFlow do it for you automatically, but now you know what they're doing.

> **بالعربي:** قاعدة السلسلة (chain rule) بتقول: لو عندك دوال جوّا بعض (زي طبقات الشبكة العصبية)، اضرب ميول كل طبقة في بعض عشان تعرف تأثير المدخل على المخرج النهائي. الـ backpropagation اللي بتسمع عنه هو حرفيًا تطبيق القاعدة دي من آخر طبقة لأول طبقة.

### 3.6 Gradient descent — calculus at work

**Intuition:** you're on a foggy hill and want to reach the valley. You can't see far, but you can feel the slope under your feet. So you take a small step downhill, feel again, and repeat.

The update rule:

```latex
w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial L}{\partial w}
```

- L — the loss (how wrong the model is)
- α — the **learning rate** (step size)

Worked example: minimize L(w) = (w − 4)². The best answer is w = 4. The derivative is L'(w) = 2(w − 4). Start at w = 0 with α = 0.25.

| Step | w | L'(w) | New w = w − 0.25 × L'(w) |
| --- | --- | --- | --- |
| 1 | 0 | −8 | 2 |
| 2 | 2 | −4 | 3 |
| 3 | 3 | −2 | 3.5 |
| 4 | 3.5 | −1 | 3.75 |

Each step halves the distance to 4. The model is learning.

Picking the learning rate:

- Too small → learning is painfully slow.
- Too large → you overshoot the valley and may never settle.

> **بالعربي:** الـ gradient descent زي إنك واقف على تل في الضباب ومش شايف الوادي. بتحسّ الأرض تحت رجلك، وبتمشي خطوة صغيرة في اتجاه النزول. الخطوة (learning rate) لو كبيرة أوي هتتقفّز فوق الوادي ومش هتنزل، ولو صغيرة أوي هتمشي ببطء شديد. الموديل بيكرر الخطوة دي آلاف المرات لحد ما الخطأ يقل.

### Practice — Chapter 3

1. Find the derivative of f(x) = 5x³ + 2x − 9.
2. Find the derivative of f(x) = 4eˣ + ln(x).
3. For f(x, y) = 3x² + xy + y², find ∂f/∂x and ∂f/∂y.
4. Compute the gradient of that same f at the point (1, 1).
5. Use the chain rule: find dy/dx for y = (2x − 5)³.
6. Minimize L(w) = (w − 10)² starting at w = 0 with α = 0.1. What is w after two steps?

## Chapter 4: Probability

Probability is the math of uncertainty. ML models rarely say "this is a cat"; they say "I'm 92% sure this is a cat", and that's a probability.

> **بالعربي:** الاحتمالات هي رياضة عدم اليقين. الموديل نادرًا بيقول "ده قطة" — هو بيقول "أنا متأكد 92% إن ده قطة"، والرقم ده احتمال.

### 4.1 The basics

**Intuition:** probability is a number from 0 (impossible) to 1 (certain).

```latex
P(A) = \frac{\text{number of ways A can happen}}{\text{total number of outcomes}}
```

Worked example: roll a fair die. P(even) = 3/6 = 0.5, because 2, 4 and 6 are even.

Three rules to know:

- **Complement:** P(not A) = 1 − P(A). If P(rain) = 0.3, then P(no rain) = 0.7.
- **AND, independent events:** P(A and B) = P(A) × P(B). Two coin flips both heads: 0.5 × 0.5 = 0.25.
- **OR:** P(A or B) = P(A) + P(B) − P(A and B). Subtract the overlap so you don't count it twice.

> **بالعربي:** الاحتمال رقم من 0 (مستحيل) لـ 1 (مؤكد)، وبيتحسب: عدد الحالات اللي تعجبك ÷ كل الحالات الممكنة. والقواعد التلاتة:
>
> - **المكمّل:** P(مش A) = 1 − P(A). لو احتمال المطر 0.3، يبقى احتمال مافيش مطر 0.7.
> - **و (AND) للأحداث المستقلة:** اضرب. وش ووش في عملتين = 0.5 × 0.5 = 0.25.
> - **أو (OR):** اجمع الاحتمالين واطرح التداخل، عشان ماتعدّش نفس الحالة مرتين.

### 4.2 Conditional probability

**Intuition:** the probability of A once you already know B happened. The "|" is read as "given".

```latex
P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}
```

Worked example: in a class of 100 students, 40 play football, and 10 of those also play chess.

- P(chess | football) = 10 / 40 = 0.25

Knowing someone plays football shrinks your "world" to those 40 students.

> **بالعربي:** الاحتمال الشرطي (conditional probability) يعني: "لو عرفت إن حاجة حصلت، إيه احتمال حاجة تانية؟" بدل ما تبصّ على كل الطلاب، بتبصّ بس على اللي بيلعبوا كورة وتسأل: "منهم كام بيلعب شطرنج؟" — عالمك اتقلّص.

### 4.3 Bayes' theorem

**Intuition:** Bayes lets you flip a conditional probability around and update your belief when new evidence arrives.

```latex
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
```

Worked example — a spam filter:

- 20% of emails are spam: P(spam) = 0.2
- 60% of spam emails contain the word "free": P(free | spam) = 0.6
- 5% of normal emails contain "free": P(free | not spam) = 0.05

Step 1, how often does "free" appear at all?

- P(free) = 0.6 × 0.2 + 0.05 × 0.8 = 0.12 + 0.04 = 0.16

Step 2, apply Bayes:

- P(spam | free) = (0.6 × 0.2) / 0.16 = 0.12 / 0.16 = **0.75**

Seeing "free" raises the spam chance from 20% to 75%.

> **بالعربي:** نظرية بايز بتقلب السؤال. بدل ما تسأل "لو الإيميل سبام، إيه احتمال يكون فيه كلمة free؟" بتسأل "لو فيه كلمة free، إيه احتمال يكون سبام؟" بتبدأ باعتقاد أولي (20% سبام) وبتحدّثه لما تشوف دليل جديد (كلمة free). الناتج: اعتقادك اتغير من 20% لـ 75%. ده بالظبط اللي بيعمله مصنّف Naive Bayes.

**Why ML cares:** this is exactly how the Naive Bayes classifier works. Bayesian thinking also underlies probabilistic models and uncertainty estimates.

### 4.4 Random variables

**Intuition:** a random variable is a number whose value depends on chance, like the result of a die roll or tomorrow's temperature.

- **Discrete:** countable values (die rolls, number of clicks).
- **Continuous:** any value in a range (height, price, time).

**Expected value** is the long-run average:

```latex
E[X] = \sum x_i \cdot P(x_i)
```

Worked example, a fair die: E\[X\] = (1 + 2 + 3 + 4 + 5 + 6) × 1/6 = 21/6 = **3.5**.

> **بالعربي:** المتغير العشوائي (random variable) رقم قيمته بتعتمد على الحظ، زي نتيجة الزار أو حرارة بكره.
>
> - **منفصل (discrete):** قيم معدودة — الزار، عدد الكليكات.
> - **متصل (continuous):** أي قيمة في مدى — الطول، السعر، الوقت.
> - **القيمة المتوقعة (expected value):** اضرب كل قيمة في احتمالها واجمع. الزار بيطلع 3.5، رغم إن مفيش وش فيه 3.5 — لأنها متوسط على المدى الطويل.

### 4.5 Distributions you'll meet

A distribution describes how likely each value is.

| Distribution | What it models | ML example |
| --- | --- | --- |
| Bernoulli | one yes/no trial (probability p) | is this email spam? |
| Binomial | number of successes in n yes/no trials | clicks out of 100 ads shown |
| Uniform | every value equally likely | random weight initialization |
| Normal (Gaussian) | the bell curve: most values near the mean | heights, measurement noise, errors |
| Poisson | count of events in a fixed time | website visits per hour |
| Categorical | one of several classes | the softmax output of a classifier |

**The normal distribution** is the star. It is described by just two numbers: the mean μ (the center) and the standard deviation σ (the spread).

The 68–95–99.7 rule:

- About 68% of values fall within 1σ of the mean.
- About 95% fall within 2σ.
- About 99.7% fall within 3σ.

Worked example: adult heights with μ = 170 cm and σ = 10 cm. About 95% of people are between 150 cm and 190 cm.

**Why ML cares:** many models assume errors are normally distributed. Feature standardization (making features mean 0, standard deviation 1) is built on the same idea.

> **بالعربي:** التوزيع (distribution) بيوصف كل قيمة احتمالها أد إيه:
>
> - **Bernoulli:** تجربة واحدة نعم/لأ — الإيميل ده سبام ولا لأ.
> - **Binomial:** عدد النجاحات في n تجربة — كام كليك من 100 إعلان.
> - **Uniform:** كل القيم بنفس الاحتمال — تهيئة الأوزان العشوائية.
> - **Normal (الجرس):** أغلب القيم حوالين المتوسط — الأطوال، الضوضاء، الأخطاء.
> - **Poisson:** عدد الأحداث في وقت محدد — زيارات الموقع في الساعة.
> - **Categorical:** واحد من كذا تصنيف — مخرجات الـ softmax.
>
> والتوزيع الطبيعي بيتوصف برقمين بس: المتوسط μ (المركز) والانحراف المعياري σ (التفرقع). وقاعدة 68–95–99.7 بتقول: 68% من القيم جوه σ واحد من المتوسط، 95% جوه 2σ، و 99.7% جوه 3σ.

### Practice — Chapter 4

1. A bag has 3 red and 7 blue balls. What is P(red)? What is P(not red)?
2. You flip a fair coin 3 times. What is P(all heads)?
3. Out of 200 customers, 80 bought a phone and 20 of those also bought a case. Find P(case | phone).
4. A disease affects 1% of people. A test catches 90% of sick people but also flags 10% of healthy people. If you test positive, what is the chance you are sick?
5. A game pays $10 with probability 0.2 and $0 otherwise. What is the expected payout?
6. Exam scores are normal with μ = 70 and σ = 5. What range holds about 68% of scores?

## Chapter 5: Statistics

Statistics turns raw data into understanding. Before you train any model, you describe the data; after you train it, you check whether the results are real or just luck.

We'll use one small dataset for the whole chapter: daily coffee sales of **\[4, 8, 6, 5, 7\]** cups.

> **بالعربي:** الإحصاء بيحوّل البيانات الخام لفهم. قبل ما تدرّب أي موديل بتوصف البيانات الأول، وبعد ما تدرّبه بتتأكد إن النتيجة حقيقية ولا مجرد حظ.

### 5.1 Mean, median, mode

**Intuition:** three ways to answer "what's a typical value?"

- **Mean (average):** add them up, divide by the count. (4 + 8 + 6 + 5 + 7) / 5 = 30 / 5 = **6**.
- **Median:** sort the values and take the middle one. Sorted: \[4, 5, 6, 7, 8\], median = **6**.
- **Mode:** the most frequent value. Here every value appears once, so there is no mode.

```latex
\mu = \frac{1}{n} \sum_{i=1}^{n} x_i
```

**Mean vs median:** one extreme value (an outlier) drags the mean but barely moves the median. Salaries \[3, 4, 5, 100\] have mean 28 but median 4.5. Use the median when data has outliers.

> **بالعربي:** تلات طرق تجاوب على "إيه القيمة الطبيعية هنا؟"
>
> - **المتوسط (mean):** اجمع كله واقسم على العدد.
> - **الوسيط (median):** رتّب الأرقام وخُد اللي في النص.
> - **المنوال (mode):** الرقم اللي بيتكرر أكتر (وساعات مايكونش موجود).
>
> والفرق المهم: رقم شاذ واحد (outlier) بيجرجر المتوسط بعيد ومابيحرّكش الوسيط تقريبًا. رواتب \[3, 4, 5, 100\] متوسطها 28 ووسيطها 4.5 — الوسيط هو اللي بيوصف الواقع.

### 5.2 Variance and standard deviation

**Intuition:** how spread out the data is. Are the values huddled near the mean or scattered far away?

```latex
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2 \qquad \sigma = \sqrt{\sigma^2}
```

Worked example with \[4, 8, 6, 5, 7\] and mean 6:

| Value | Value − mean | Squared |
| --- | --- | --- |
| 4 | −2 | 4 |
| 8 | 2 | 4 |
| 6 | 0 | 0 |
| 5 | −1 | 1 |
| 7 | 1 | 1 |

- Variance = (4 + 4 + 0 + 1 + 1) / 5 = 10 / 5 = **2**
- Standard deviation = √2 ≈ **1.41** cups

Why square? So negative and positive gaps don't cancel out. Standard deviation takes the square root to get back to the original units.

Note: for a *sample*, divide by n − 1 instead of n. NumPy's `np.var` divides by n by default; use `ddof=1` for the sample version.

**Why ML cares:** standardizing features uses these directly: z = (x − μ) / σ. It puts every feature on the same scale so no feature dominates just because its numbers are bigger.

> **بالعربي:** التباين (variance) بيقيس الأرقام متفرقة عن المتوسط أد إيه. لو كلها قريبة من المتوسط، التباين صغير. لو متنطّطة بعيد، التباين كبير. الانحراف المعياري (standard deviation) هو جذر التباين — بيرجّع الوحدة الأصلية. الـ standardization (z-score) بتحوّل أي فيتشر لمتوسط 0 وانحراف 1، عشان مايبقاش فيتشر مسيطر على الموديل لمجرد إن أرقامه كبيرة.

### 5.3 Covariance and correlation

**Intuition:** do two variables move together?

- **Covariance:** positive if they rise together, negative if one rises while the other falls. Its size is hard to read because it depends on units.
- **Correlation (r):** covariance rescaled to always lie between −1 and 1.

```latex
r = \frac{\text{cov}(X, Y)}{\sigma_X \, \sigma_Y}
```

How to read r:

| r | Meaning | Example |
| --- | --- | --- |
| close to 1 | strong positive link | hours studied and exam score |
| close to 0 | no linear link | shoe size and exam score |
| close to −1 | strong negative link | car age and resale price |

**Golden rule:** correlation is not causation. Ice cream sales and drownings rise together, but hot weather causes both.

**Why ML cares:** correlation helps you pick useful features and spot redundant ones. Two features with r ≈ 0.99 carry almost the same information.

> **بالعربي:** السؤال هنا: الحاجتين بيتحركوا مع بعض ولا لأ؟
>
> - **التغاير (covariance):** موجب لو الاتنين بيطلعوا مع بعض، سالب لو واحد يطلع والتاني ينزل. بس حجمه صعب تقراه لأنه بيعتمد على الوحدات.
> - **الارتباط (correlation أو r):** نفس الفكرة بس مقيّسة ما بين −1 و 1، فتقدر تقراها على طول: قريب من 1 علاقة طرديّة قوية، قريب من 0 مفيش علاقة خطية، قريب من −1 علاقة عكسيّة قوية.
> - **القاعدة الذهبية:** الارتباط مش معناه السببية. مبيعات الآيس كريم وحالات الغرق بيطلعوا مع بعض، بس السبب الحقيقي في الاتنين هو الجو الحرّ.
>
> وفي ML: فيتشرين بينهم r ≈ 0.99 بيحملوا نفس المعلومة تقريبًا، فواحد منهم زيادة.

### 5.4 Samples and populations

**Intuition:** you can rarely measure everyone (the **population**), so you measure some of them (a **sample**) and generalize.

- A good sample is **random** and **representative**. Survey only your friends and you'll get biased answers.
- Bigger samples give more reliable estimates.
- **Central limit theorem:** averages of many random samples follow a normal distribution, even when the original data doesn't. This is why the bell curve shows up everywhere.

**Why ML cares:** your training data is a sample of the real world. Your test set is another sample. If they don't represent real users, your model will disappoint in production.

> **بالعربي:**
>
> - **المجتمع (population)** هو كل الناس، و**العيّنة (sample)** هي اللي قدرت تقيسهم فعلًا.
> - العيّنة الكويسة **عشوائية** و**ممثّلة**. لو سألت أصحابك بس، النتيجة هتبقى متحيّزة.
> - العيّنة الأكبر بتدّيك تقدير أدق.
> - **نظرية النهاية المركزية (central limit theorem):** متوسطات العيّنات العشوائية بتتوزّع توزيع طبيعي، حتى لو البيانات الأصلية مش طبيعية — عشان كده منحنى الجرس بيطلعلك في كل حاجة.
>
> والمهم في ML: بيانات التدريب عيّنة من الدنيا الحقيقية. لو مش ممثّلة لليوزرز الحقيقيين، الموديل هيخيّب أملك في الـ production.

### 5.5 Hypothesis testing

**Intuition:** a courtroom for data. You assume "nothing special is happening" (innocent) until the evidence is strong enough to say otherwise.

The steps:

1. **Null hypothesis (H₀):** no effect. Example: "the new website design doesn't change sales."
2. **Alternative hypothesis (H₁):** there is an effect.
3. Collect data and compute a test statistic.
4. Get the **p-value**: the chance of seeing results at least this extreme *if H₀ were true*.
5. If p < 0.05 (the usual threshold), reject H₀. The result is "statistically significant".

Worked example: an A/B test gives p = 0.01 for the new design. If the design truly had no effect, results this strong would appear only 1% of the time. So you reject H₀ and conclude the design likely helps.

Two kinds of mistakes:

- **Type I error (false positive):** you claim an effect that isn't real.
- **Type II error (false negative):** you miss an effect that is real.

**Why ML cares:** you'll use this to check whether model A is truly better than model B, and in A/B tests when a model goes live. False positives and false negatives also return as precision and recall.

> **بالعربي:** اختبار الفرضيات زي المحكمة: المتهم بريء لحد ما يثبت العكس. الـ H₀ (الفرضية الصفرية) بتقول "مفيش فرق" أو "مفيش تأثير". بتجمع بيانات وتحسب الـ p-value: لو صغير (أقل من 0.05)، يبقى الدليل قوي كفاية إنك ترفض H₀. النوعين من الغلط: Type I (قلت فيه فرق وهو مفيش — false positive) و Type II (قلت مفيش فرق وهو فيه — false negative). نفس الفكرة بتظهر في precision و recall.

### 5.6 Statistics in Python

```python
import numpy as np
x = np.array([4, 8, 6, 5, 7])
np.mean(x), np.median(x)   # 6.0, 6.0
np.var(x), np.std(x)       # 2.0, 1.414
np.corrcoef(x, y)          # correlation with another array y
```

> **بالعربي:** كل حسابات الشابتر بسطر واحد لكل واحدة في NumPy. خلي بالك بس من `np.var`: بيقسم على n بشكل افتراضي، فلو إنت بتحسب لعيّنة (sample) استخدم `ddof=1` عشان يقسم على n − 1.

### Practice — Chapter 5

1. Find the mean and median of \[2, 3, 3, 5, 12\]. Which better describes a typical value, and why?
2. Find the variance and standard deviation of \[1, 3, 5\].
3. A feature has mean 50 and standard deviation 10. Standardize the value 65.
4. Two features have r = −0.9. Describe their relationship in plain words.
5. A test comparing two models gives p = 0.30. What do you conclude?
6. A spam filter marks an important email as spam. Is that a Type I or Type II error, if H₀ is "this email is not spam"?

## Chapter 6: Putting it together — linear regression from scratch

This chapter trains a real model by hand, using every chapter you've read. Statistics measures the error, calculus finds which way to fix it, and linear algebra does it for many examples at once.

> **بالعربي:** هنا هندرّب موديل حقيقي بإيدينا، وهنستخدم كل اللي قريناه: الإحصاء بيقيس الخطأ، التفاضل بيقولنا نمشي فين، والجبر الخطي بيعمل ده لكل الأمثلة مرة واحدة.

### 6.1 The problem

Hours studied (x) and exam score in tens of points (y):

| Student | Hours (x) | Score (y) |
| --- | --- | --- |
| A | 1 | 2 |
| B | 2 | 4 |
| C | 3 | 6 |

We want a model **ŷ = w·x** that predicts the score from the hours. By eye the answer is w = 2, but the model must find it by itself.

> **بالعربي:** المسألة: عندنا ساعات مذاكرة (x) ودرجات (y)، وعايزين موديل ŷ = w·x يتوقّع الدرجة من الساعات. بالعين إنت شايف إن w = 2، بس الفكرة كلها إن الموديل لازم يوصلها بنفسه من غير ما حد يقوله.

### 6.2 Step 1 — measure the error (statistics)

We use the **mean squared error (MSE)**: the average of the squared gaps between prediction and truth. It is the variance formula from Chapter 5, applied to errors.

```latex
L(w) = \frac{1}{n} \sum_{i=1}^{n} (w x_i - y_i)^2
```

At w = 0 every prediction is 0, so L = (2² + 4² + 6²) / 3 = 56 / 3 ≈ **18.67**. That's bad.

> **بالعربي:** الـ MSE (متوسط مربع الخطأ) بيقيس الموديل غلطان أد إيه: خُد الفرق بين التوقّع والحقيقة، ربّعه (عشان السالب والموجب مايلغوش بعض)، وخُد المتوسط. ودي نفس صيغة التباين من شابتر 5 بس مطبّقة على الأخطاء. عند w = 0 الخطأ ≈ 18.67 — وحش جدًا، وطبيعي لأن الموديل لسه مابيعرفش حاجة.

### 6.3 Step 2 — find the slope of the error (calculus)

Using the power rule and the chain rule from Chapter 3:

```latex
\frac{dL}{dw} = \frac{2}{n} \sum_{i=1}^{n} (w x_i - y_i) \cdot x_i
```

At w = 0: (2/3) × \[(0 − 2)·1 + (0 − 4)·2 + (0 − 6)·3\] = (2/3) × (−28) ≈ **−18.67**.

The slope is negative, so the error goes down if we increase w.

> **بالعربي:** بنشتق الخطأ بالنسبة لـ w عشان نعرف نمشي فين. طلعت المشتقة ≈ −18.67، والسالب هنا معناه: "لو كبّرت w، الخطأ هينزل". يعني التفاضل حرفيًا بيشاورلك على الطريق الصح.

### 6.4 Step 3 — take a step (gradient descent)

Update with learning rate α = 0.05:

```latex
w_{\text{new}} = w - 0.05 \cdot \frac{dL}{dw}
```

Repeating the update:

| Step | w | Loss L(w) |
| --- | --- | --- |
| 0 | 0.000 | 18.67 |
| 1 | 0.933 | 5.31 |
| 2 | 1.431 | 1.51 |
| 3 | 1.697 | 0.43 |
| 4 | 1.838 | 0.12 |

Each step closes about half the remaining gap to w = 2, and the loss drops toward 0. That is learning.

> **بالعربي:** بنحدّث w بخطوة صغيرة في الاتجاه العكسي للمشتقة (α = 0.05). كل خطوة بتقرّب نص المسافة الباقية لـ w = 2، والخطأ بينزل من 18.67 لـ 0.12 في 4 خطوات بس. ده التعلّم نفسه، مفيش أي سحر.

### 6.5 Step 4 — do it for all examples at once (linear algebra)

Instead of looping over students, stack them into vectors and use dot products:

```latex
\hat{\mathbf{y}} = X\mathbf{w} \qquad \nabla L = \frac{2}{n} X^T (X\mathbf{w} - \mathbf{y})
```

The transpose (Chapter 2) lines up the shapes. With many features and a million rows, this is still one line of code.

> **بالعربي:** بدل ما تلفّ على كل طالب لوحده، رصّهم في مصفوفة واحدة واضرب. الـ transpose (من شابتر 2) بيخلّي الأشكال تتطابق. والجميل إن السطرين دول مابيتغيروش لو عندك مليون صف و 500 فيتشر — نفس الكود بالظبط، وده سبب وجود الـ GPU.

### 6.6 The whole thing in Python

```python
import numpy as np

X = np.array([[1], [2], [3]], dtype=float)   # hours
y = np.array([2, 4, 6], dtype=float)         # scores
w = np.zeros(1)                              # start at w = 0
alpha = 0.05

for step in range(50):
    y_hat = X @ w                             # linear algebra: predictions
    loss = np.mean((y_hat - y) ** 2)          # statistics: MSE
    grad = (2 / len(y)) * X.T @ (y_hat - y)   # calculus: gradient
    w = w - alpha * grad                      # gradient descent step

print(w)   # about [2.0]
```

Every deep learning model, from image classifiers to large language models, follows this same loop: predict, measure the error, compute the gradient, step. Only the size changes.

> **بالعربي:** كل حاجة في الشابتر ده اتجمّعت هنا. الإحصاء (MSE) بيقيسلك الخطأ، التفاضل (gradient) بيقولك تمشي فين عشان الخطأ يقل، والجبر الخطي (matrix multiplication) بيخلّيك تعمل ده لكل الأمثلة مرة واحدة بدل ما تلفّ عليهم واحد واحد. اللوب ده — predict, measure, gradient, step — هو اللي بيحصل في أي موديل من أبسط linear regression لحد GPT.

### 6.7 Try it yourself

1. Change alpha to 0.2. What happens to w? Then try 0.01.
2. Add a bias b so the model is ŷ = w·x + b, and use the data y = \[3, 5, 7\]. Both w and b need their own partial derivative.
3. Print the loss every 10 steps and confirm it keeps falling.

> **بالعربي:** جرّب بإيدك: كبّر الـ alpha لـ 0.2 وشوف w بتتقفّز إزاي، وصغّرها لـ 0.01 وشوف بتبطّأ إزاي. بعدين ضيف الـ bias وخلي الموديل ŷ = w·x + b — هتحتاج مشتقة جزئية لكل من w و b، ودي شابتر 3 بالظبط.

## Answer key

**Chapter 1**

1. f(4) = 3 × 4 − 2 = 10
2. 2⁴ × 2² = 2⁶ = 64
3. log₂(32) = 5, because 2⁵ = 32
4. 1² + 2² + 3² = 1 + 4 + 9 = 14
5. σ(0) = 1 / (1 + e⁰) = 1 / 2 = 0.5

**Chapter 2**

1. \[3, 3, 3\]
2. √(36 + 64) = √100 = 10
3. 1×3 + 0×5 + 2×4 = 11
4. Yes. The dot product is 1×(−2) + 2×1 = 0.
5. A × B is 4 × 2. B × A is not possible: B is 3 × 2 and A is 4 × 3, and the inner numbers (2 and 4) don't match.
6. \[\[3, 1\], \[6, 4\]\]
7. 4×1 − 2×3 = −2. It's not zero, so the inverse exists.

**Chapter 3**

1. f'(x) = 15x² + 2
2. f'(x) = 4eˣ + 1/x
3. ∂f/∂x = 6x + y and ∂f/∂y = x + 2y
4. At (1, 1): ∇f = \[7, 3\]
5. dy/dx = 3(2x − 5)² × 2 = 6(2x − 5)²
6. L'(w) = 2(w − 10). Step 1: w = 0 − 0.1 × (−20) = 2. Step 2: w = 2 − 0.1 × (−16) = **3.6**.

**Chapter 4**

1. P(red) = 3/10 = 0.3 and P(not red) = 0.7
2. 0.5 × 0.5 × 0.5 = 0.125
3. P(case | phone) = 20/80 = 0.25
4. P(positive) = 0.9 × 0.01 + 0.1 × 0.99 = 0.108. P(sick | positive) = 0.009 / 0.108 ≈ **8.3%**. Surprisingly low, because healthy people far outnumber sick ones.
5. E = 10 × 0.2 + 0 × 0.8 = $2
6. 65 to 75 (μ ± 1σ)

**Chapter 5**

1. Mean = 25/5 = 5, median = 3. The median is better: the outlier 12 pulls the mean up.
2. Mean = 3. Squared gaps: 4, 0, 4. Variance = 8/3 ≈ 2.67, standard deviation ≈ 1.63.
3. z = (65 − 50) / 10 = 1.5
4. Strong negative relationship: when one goes up, the other usually goes down.
5. p = 0.30 is above 0.05, so you can't reject H₀. There's no solid evidence that one model is better.
6. Type I error (false positive): you rejected H₀ ("not spam") when it was true.

## Cheat sheet

| Topic | Key formula | Plain words |
| --- | --- | --- |
| Dot product | a · b = Σ aᵢbᵢ | multiply matching entries, add up |
| Vector length | ‖x‖ = √(Σ xᵢ²) | Pythagoras in many dimensions |
| Matrix shapes | (m × n)(n × p) → (m × p) | inner numbers must match |
| Linear model | ŷ = Xw + b | weights · features + bias |
| Power rule | (xⁿ)' = n·xⁿ⁻¹ | bring the power down, reduce it by 1 |
| Chain rule | dy/dx = dy/du · du/dx | multiply the slopes of stacked functions |
| Gradient descent | w ← w − α·∇L | step downhill, size α |
| Conditional probability | P(A\|B) = P(A and B) / P(B) | probability within a smaller world |
| Bayes | P(A\|B) = P(B\|A)·P(A) / P(B) | update a belief with evidence |
| Expected value | E\[X\] = Σ xᵢP(xᵢ) | long-run average |
| Variance | σ² = (1/n)Σ(xᵢ − μ)² | average squared distance from the mean |
| Standardize | z = (x − μ) / σ | how many σ away from the mean |
| MSE loss | (1/n)Σ(ŷᵢ − yᵢ)² | average squared prediction error |

> **بالعربي:** الجدول ده هو الكتاب كله في صفحة واحدة. اطبعه وعلّقه قصادك، وكل ما تنسى صيغة بصّ هنا بدل ما ترجع للشابتر من الأول.

## 8-week study plan

About 45 minutes a day, 5 days a week.

| Week | Focus | Goal by the end of the week |
| --- | --- | --- |
| 1 | Chapter 1: warm-up | read notation, use logs and Σ comfortably |
| 2 | Chapter 2: vectors and dot products (2.1–2.2) | compute dot products and lengths by hand and in NumPy |
| 3 | Chapter 2: matrices (2.3–2.8) | multiply matrices and predict result shapes |
| 4 | Chapter 3: derivatives and rules (3.1–3.3) | differentiate polynomials and partial derivatives |
| 5 | Chapter 3: gradient, chain rule, gradient descent (3.4–3.6) | run gradient descent by hand for 3 steps |
| 6 | Chapter 4: probability | solve a Bayes problem from scratch |
| 7 | Chapter 5: statistics | describe a dataset and interpret a p-value |
| 8 | Chapter 6 + review | code linear regression from memory |

**Want more after this book?** Good free follow-ups are 3Blue1Brown's "Essence of Linear Algebra" and "Essence of Calculus" video series, Khan Academy's statistics course, and the book *Mathematics for Machine Learning* by Deisenroth, Faisal and Ong (free PDF from the authors).

> **بالعربي:** الخطة 45 دقيقة في اليوم، 5 أيام في الأسبوع، 8 أسابيع. والمهم الاستمرار مش الطول: نص ساعة كل يوم أحسن بكتير من 5 ساعات مرة في الأسبوع. ومعيار النجاح في كل أسبوع مش إنك "قريت" — إنك تقدر تحل بإيدك وتشرح الفكرة لحد تاني.
