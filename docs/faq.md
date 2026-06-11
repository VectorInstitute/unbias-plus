# FAQ

Common questions about how UnBias-Plus works, what it does and does not do, and how to use it safely. For technical details, see [How it works](how_it_works.md). For installation and usage, see [Installation](guides/install.md).

---

## What is UnBias-Plus?

Vector Institute's UnBias-Plus is a free, open-source bias detection toolkit purpose-built for the general public. Created by Vector's AI Engineering team, the tool takes any piece of text, identifies biased language, explains why specific phrases are problematic, suggests neutral alternatives, and returns a fully rewritten version. The web demo runs in a browser with no installation required, while a developer platform (Python package, CLI, and REST API) makes it usable for researchers and data scientists.

## Data privacy

### What happens to text I submit through the demo?

When you submit text through Vector's UnBias-Plus web demo, nothing is stored or shared. Text submitted through the browser tool is processed in real time and discarded immediately after the bias analysis. Vector does not retain submissions, review them, or use them to train our models. No account, login, or personal information is required to use the tool. For sensitive or unpublished material, we recommend self-hosting our open-source Python package, which runs entirely on your own machine with no data ever leaving your device.

---

## About UnBias-Plus

### What kinds of bias does UnBias-Plus detect?

UnBias-Plus identifies several categories of biased language:

- **Loaded language** &mdash; words with strong emotional connotations
- **Dehumanizing framing** &mdash; language that strips dignity from groups
- **False generalizations** &mdash; sweeping statements like *"they always"* or *"all of them"*
- **Framing bias** &mdash; selective wording that implies a viewpoint
- **Euphemism and dysphemism** &mdash; softening or hardening language to shape perception
- **Politically charged terminology** &mdash; labels used to provoke rather than describe
- **Sensationalism** &mdash; exaggerated language meant to trigger emotional reactions

Each flagged segment is rated by severity (low, medium, or high) and accompanied by a short explanation. UnBias-Plus was trained primarily on English news articles, so the tool works best on journalistic and news-style text. It can surface common patterns like gender bias, racial stereotypes, hiring language, and workplace framing. Results should always be reviewed by a human, since bias is context-dependent.

### How is content determined to be biased or unbiased?

Vector built UnBias-Plus on language models fine-tuned with real news data and expert bias annotations, refined through multiple human-in-the-loop iterations. For each piece of text, the model identifies specific phrases that show signs of bias, classifies the type of bias (such as loaded language, framing, or sensationalism), assigns a severity rating, and provides a short explanation along with a neutral rewrite.

Like any tool, UnBias-Plus is not infallible. Bias is subjective and context-dependent, so results should be treated as indicative rather than definitive. UnBias-Plus is designed to support editorial judgment, not replace it.

We welcome scrutiny and user feedback as part of our ongoing research and commitment to refine the tool based on real-world use. For more details on the methodology, training data, and model architecture, see our [GitHub repository](https://github.com/VectorInstitute/unbias-plus/).

### What data is the model trained on?

UnBias-Plus is fine-tuned on a curated dataset of English-language news articles assembled by the Vector Institute. Each article in the training set includes expert-annotated bias labels, segment-level annotations identifying biased phrases, and human-written neutral rewrites that preserve the original meaning.

The dataset spans a broad range of news topics and outlets to reflect real-world reporting styles, and was reviewed through multiple human-in-the-loop iterations to improve labeling consistency. Because the training data is news-focused and primarily English, the model performs best on journalistic text. For dataset and model details, see the [model card on Hugging Face](https://huggingface.co/vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct-Legacy).

### Does UnBias-Plus detect misinformation or AI-generated content?

No. UnBias-Plus focuses specifically on detecting biased language in text. It does not verify factual accuracy, fact-check claims, or identify AI-generated content. These are separate and important challenges that Vector's research community continues to work on through related projects.

To stay informed about upcoming tools, including our work on misinformation detection and multimodal (audio and visual) bias analysis, sign up for the [Vector Institute newsletter](https://vectorinstitute.ai/) or follow our [ongoing work](https://vectorinstitute.github.io/vector-aixpert/).

### Does UnBias-Plus work in French or other languages?

UnBias-Plus is currently trained and optimized for English text. It may produce results when given content in other languages, but accuracy and reliability cannot be guaranteed outside of English. Expanding to French and additional languages is on the roadmap, and feedback from multilingual users is welcome.

### Is there a word or character limit?

The web demo accepts up to 5,000 characters per submission, which is suitable for short articles, paragraphs, or excerpts. The model performs best on news-style articles under that limit.

For longer documents, full articles, or batch processing, install the open-source package directly. It supports the command line, a Python API, and a self-hosted REST API, giving you full control over input length and processing volume. See [Installation](guides/install.md) to get started.

!!! tip "Working with long documents"
    The default model targets article-length input. For long or dense
    text, use the larger model, or chunk the document, so coverage isn't lost to truncation.

---

## Open-source code

### Where can I access the code?

The full UnBias-Plus codebase is available on [GitHub](https://github.com/VectorInstitute/unbias-plus). It is built for researchers, developers, journalists, and educators who want to integrate bias detection into newsroom tools, academic research, content platforms, hiring reviews, or media literacy projects. The tool can be used as a Python package, a self-hosted REST API, or a command-line interface. See [Installation](guides/install.md) for installation and usage.

!!! note "On third-party use"

    UnBias-Plus is released as open source in the spirit of transparency and collaborative improvement. The Vector Institute cannot control downstream use of the code, and any modified or third-party versions are not affiliated with, nor endorsed by, the Vector Institute.

---

## Feedback

Share your experience through our [feedback form](https://docs.google.com/forms/d/e/1FAIpQLSfmYHzgOYBvWMvNiumfP8Xql-2VUNkmu8fw6LsZ-0evNQ8eSQ/viewform?usp=header) &mdash; it takes about a minute. For bug reports and feature requests, please use [GitHub Issues](https://github.com/VectorInstitute/unbias-plus/issues).
