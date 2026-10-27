# أساسيات الرياضيات لتعلُّم الآلة — ببساطة

كتاب عربي قصير وعملي في التلات مجالات اللي محتاجها فعلاً في تعلُّم الآلة:
**الجبر الخطي**، و**التفاضل**، و**الاحتمالات والإحصاء**.

**اقرا الكتاب هنا: https://astromar96.github.io/ml-math-book/**

الكتاب مكتوب لأي حد ماشي في طريقه لوظيفة junior ML engineer وخلّص رياضة ثانوية
(أو نسي أغلبها). مفيش براهين ومفيش نظريات تقيلة. كل موضوع ماشي بنفس الترتيب:
الحدس بالكلام العادي ← الصيغة ← مثال محلول بأرقام ← ليه ML محتاجة ده ← تمارين
وحلولها في آخر الكتاب. والمصطلحات الإنجليزية مكتوبة جنب العربي عشان تعرفها في
الكود والـ papers.

## المحتويات

| الفصل | الموضوع |
| --- | --- |
| 1 | تسخين رياضي: الرموز، الدوال، الأسس، اللوغاريتمات، Σ |
| 2 | الجبر الخطي: المتجهات، الضرب النقطي، التشابه الجيبي، المصفوفات، المنقول، المعكوس، المتجهات الذاتية |
| 3 | التفاضل: المشتقات، المشتقات الجزئية، التدرج، قاعدة السلسلة، النزول التدرجي وأنواعه |
| 4 | الاحتمالات: الأساسيات، الاحتمال الشرطي، بايز، التوزيعات، softmax، cross-entropy، الأرجحية العظمى |
| 5 | الإحصاء: المتوسط والتباين، الارتباط، فترات الثقة، اختبار الفرضيات، تقسيم البيانات، الانحياز والتباين، التنظيم |
| 6 | نجمّع كله: انحدار خطي من الصفر بالورقة وبـ NumPy |
| 7 | تقييم الموديل: مصفوفة الالتباس، precision، recall، F1، ROC/AUC، مقاييس الانحدار |

وكمان مفتاح إجابات لكل التمارين، وورقة غش بكل الصيغ، وخطة مذاكرة 10 أسابيع.

## تطبيق يشتغل من غير إنترنت (PWA)

الكتاب مثبّت كتطبيق. افتح الرابط واضغط **⤓ تثبيت التطبيق** في أعلى الصفحة، أو من
قايمة المتصفح: «تثبيت» في Chrome و Edge، و«إضافة إلى الشاشة الرئيسية» من زر
المشاركة في Safari على iPhone و iPad.

بعد أول فتحة الكتاب كله بيتخزّن على جهازك — النص والمعادلات والخطوط العربية
والنسخة الإنجليزية كمان — فيشتغل كامل وأنت أوفلاين، في المترو أو الطيارة.
لما تنزل نسخة جديدة بيظهر زر **↻ نسخة جديدة** وأنت تختار إمتى تحدّث.

مفيش أي حاجة بتتحمّل من سيرفر تاني: KaTeX و marked والخطوط كلها متخزّنة جوه
المستودع في `vendor/`.

## الملفات

- `index.html` — الكتاب كامل في صفحة واحدة (المعادلات بـ KaTeX، تصميم RTL، وضع ليلي، وجاهز للطباعة)
- `book-ar.md` — المصدر بصيغة Markdown
- `english.html` + `book.md` — النسخة الإنجليزية الأقصر (6 فصول)
- `manifest.webmanifest` + `sw.js` + `icons/` — ملفات التطبيق والتخزين للعمل أوفلاين
- `vendor/` — KaTeX و marked والخطوط، متخزّنة محليًا

---

# Math Foundations for Machine Learning (English edition)

A shorter English edition (6 chapters, with Arabic explainer notes) is published
at **https://astromar96.github.io/ml-math-book/english.html** — source in `book.md`.

Both editions are one installable PWA: hit **⤓ Install** in the top bar (or "Add
to Home Screen" on iOS) and the whole book — text, formulas, fonts and the other
language edition — is cached on device and works with no network at all. Nothing
is loaded from a CDN; KaTeX, marked and the web fonts are vendored in `vendor/`.

The Arabic edition is the fuller one: it adds cosine similarity and distance
metrics, gradient-descent variants, softmax, entropy and cross-entropy, maximum
likelihood, confidence intervals, train/validation/test splits, the
bias–variance tradeoff, L1/L2 regularization, and a whole chapter on evaluation
metrics.
