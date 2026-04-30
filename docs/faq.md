# FAQ

Common questions about how UnBias-Plus works, what it does and does not do, and how to use it safely. For technical details, see [How it works](how_it_works.md). For installation and usage, see the [User Guide](user_guide.md).

---

## Data privacy

### What happens to text I submit through the demo?

Nothing is stored or shared. Text submitted through the [live demo](https://unbias-plus.vectorinstitute.ai/) is processed in real time and discarded immediately after analysis. Submissions are not retained, reviewed, or used to train the models. No account, login, or personal information is required.

For sensitive or unpublished material, we recommend self-hosting the open-source Python package. It runs entirely on your own machine, and no data leaves your device. See the [User Guide](user_guide.md) for installation.

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

Each flagged segment is rated by severity (low, medium, or high) and accompanied by a short explanation. The model was trained primarily on English news articles and works best on journalistic text, where it surfaces patterns such as gender bias, racial stereotypes, hiring language, and workplace framing. Bias is context-dependent, so results should always be reviewed by a human.

### How does UnBias-Plus decide whether something is biased?

UnBias-Plus is built on a language model fine-tuned on news data, with bias annotations refined through multiple human-in-the-loop iterations. For each piece of text, the model identifies specific phrases showing signs of bias, classifies the bias type, assigns a severity rating, and provides a short explanation alongside a neutral rewrite.

Because bias is subjective and context-dependent, results should be treated as indicative rather than definitive. UnBias-Plus is designed to support editorial judgment, not replace it. For the full pipeline and methodology, see [How it works](how_it_works.md).

### What data is the model trained on?

UnBias-Plus is fine-tuned on a curated dataset of English-language news articles assembled by the Vector Institute. Each article in the training set includes expert-annotated bias labels, segment-level annotations identifying biased phrases, and human-written neutral rewrites that preserve the original meaning.

The dataset spans a broad range of news topics and outlets to reflect real-world reporting styles, and was reviewed through multiple human-in-the-loop iterations to improve labeling consistency. Because the training data is news-focused and primarily English, the model performs best on journalistic text. For dataset and model details, see the [model card on Hugging Face](https://huggingface.co/vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct).

### Does UnBias-Plus detect misinformation or AI-generated content?

No. UnBias-Plus focuses specifically on detecting biased language in text. It does not verify factual accuracy, fact-check claims, or identify AI-generated content. These are separate and important challenges that the Vector research community continues to work on through related projects.

To stay informed about future tools, including work on misinformation detection and multimodal (audio and visual) bias analysis, follow the [Vector Institute](https://vectorinstitute.ai).

### Does UnBias-Plus work in French or other languages?

UnBias-Plus is currently trained and optimized for English text. It may produce results when given content in other languages, but accuracy and reliability cannot be guaranteed outside of English. Expanding to French and additional languages is on the roadmap, and feedback from multilingual users is welcome.

### Is there a word or character limit?

The web demo accepts up to 5,000 characters per submission, which is suitable for short articles, paragraphs, or excerpts. The model performs best on news-style articles under that limit.

For longer documents, full articles, or batch processing, install the open-source package directly. It supports the command line, a Python API, and a self-hosted REST API, giving you full control over input length and processing volume. See the [User Guide](user_guide.md) to get started.

---

## Open-source code

### Where can I access the code?

The full UnBias-Plus codebase is available on [GitHub](https://github.com/VectorInstitute/unbias-plus). It is built for researchers, developers, journalists, and educators who want to integrate bias detection into newsroom tools, academic research, content platforms, hiring reviews, or media literacy projects. The tool can be used as a Python package, a self-hosted REST API, or a command-line interface. See the [User Guide](user_guide.md) for installation and usage.

!!! note "On third-party use"

    UnBias-Plus is released as open source in the spirit of transparency and collaborative improvement. The Vector Institute cannot control downstream use of the code, and any modified or third-party versions are not affiliated with, nor endorsed by, the Vector Institute.
