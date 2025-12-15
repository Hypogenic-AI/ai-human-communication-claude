# Literature Review: AI-to-Human Communication

## Research Area Overview

This literature review examines research on making AI-to-human communication more effective, particularly when conveying large volumes of information. The key challenge is bridging the gap between AI systems (which can process dense, comprehensive information) and human cognitive capabilities (which favor concise, well-structured communication). This review covers papers on progressive disclosure, cognitive ergonomics, text summarization, evaluation metrics, and human-LLM interaction design.

---

## Key Papers

### Paper 1: Progressive Disclosure: Designing for Effective Transparency

- **Authors**: Aaron Springer, Steve Whittaker
- **Year**: 2018
- **Source**: arXiv:1811.02164
- **Key Contribution**: Introduces progressive disclosure principles for transparency in intelligent systems - showing that simpler initial feedback helps users build working mental models before receiving detailed explanations.
- **Methodology**: Two user studies comparing global vs. incremental feedback in intelligent systems. Measured user performance, preferences, and qualitative understanding.
- **Key Findings**:
  - Users initially prefer more transparent systems but retract this preference after experience
  - Incremental feedback can be distracting and undermine simple heuristics users form
  - Simplified initial feedback that hides potential errors helps users build working mental models
  - Progressive disclosure (simple first, then detailed) improves user comprehension
- **Relevance to Our Research**: Directly addresses how to present AI output effectively - suggests starting simple and progressively revealing complexity rather than presenting all information at once.

---

### Paper 2: Task Supportive and Personalized Human-Large Language Model Interaction

- **Authors**: Ben Wang, Jiqun Liu, Jamshed Karimnazarov, Nicolas Thompson
- **Year**: 2024
- **Source**: ACM CHIIR 2024, arXiv:2402.06170
- **Key Contribution**: Develops a ChatGPT-like platform with supportive functions (perception articulation, prompt suggestion, conversation explanation) that reduce cognitive load.
- **Methodology**: User study with platform integrating GPT-3.5-turbo with three supportive functions:
  1. Perception articulation - users express task expectations
  2. Prompt suggestion - system recommends better prompts
  3. Conversation explanation - system explains its reasoning
- **Datasets Used**: Custom tasks across information seeking domains
- **Key Findings**:
  - Supportive functions help users manage expectations and reduce cognitive loads
  - Prompt suggestions improve user engagement
  - Conversation explanations increase user understanding of AI reasoning
  - Benefits especially significant for underserved users with less AI experience
- **Relevance to Our Research**: Demonstrates practical methods for making LLM interactions more accessible through explanation and guidance features.

---

### Paper 3: A Comprehensive Survey on Automatic Text Summarization with LLM Methods

- **Authors**: Yang Zhang, Hanlei Jin, Dan Meng, Jun Wang, Jinghua Tan
- **Year**: 2024
- **Source**: arXiv:2403.02901
- **Key Contribution**: Comprehensive survey covering extractive, abstractive, and LLM-based summarization methods.
- **Methodology**: Survey of ATS literature from statistical methods to LLM approaches
- **Key Findings**:
  - LLMs offer paradigm-flexible summarization (can switch between extractive/abstractive)
  - In-context learning enables few-shot summarization without retraining
  - Challenges remain: hallucinations, factual accuracy, coherence
  - Evaluation methods: ROUGE, BERTScore, human evaluation, LLM-as-judge
- **Relevance to Our Research**: Provides foundation for using summarization to condense AI output for human consumption. Identifies evaluation metrics and common pitfalls.

---

### Paper 4: A Comparative Study of Quality Evaluation Methods for Text Summarization

- **Authors**: Huyen Nguyen, Haihua Chen, Lavanya Pobbathi, Junhua Ding
- **Year**: 2024
- **Source**: arXiv:2407.00747
- **Key Contribution**: Compares automatic metrics, human evaluation, and LLM-based evaluation for summarization quality.
- **Methodology**: Evaluated 7 SOTA summarization models using 8 automatic metrics plus human evaluation on patent documents.
- **Key Findings**:
  - LLM evaluation aligns more closely with human evaluation than traditional metrics
  - ROUGE-2, BERTScore, and SummaC show inconsistent alignment with human judgment
  - Proposes LLM-powered framework for automatic evaluation and improvement
- **Metrics Compared**: ROUGE, BLEU, BERTScore, MoverScore, SummaC, QuestEval
- **Relevance to Our Research**: Identifies that LLM-based evaluation is more reliable for assessing communication quality than traditional metrics.

---

### Paper 5: CogErgLLM: Cognitive Ergonomics in LLM System Design

- **Authors**: Azmine Toushik Wasi, Mst Rafia Islam
- **Year**: 2024
- **Source**: arXiv:2407.02885
- **Key Contribution**: Framework for integrating cognitive ergonomics principles into LLM design to improve safety, reliability, and user satisfaction.
- **Methodology**: Position paper providing comprehensive framework and practical guidelines
- **Key Principles for LLM Design**:
  1. Efficiency - optimize mental workload
  2. Attention support - guide user focus
  3. Learning facilitation - help users understand the system
  4. Decision-making aid - support rather than replace human judgment
  5. Adaptivity - learn user preferences over time
- **Key Findings**:
  - Current LLMs lack comprehensive cognitive ergonomics integration
  - Insufficient focus on mitigating biases through cognitive science methods
  - User-centered design principles inconsistently applied
  - Need for explainability mechanisms to increase trust
- **Relevance to Our Research**: Provides theoretical framework for designing AI-to-human communication that aligns with human cognitive capabilities.

---

### Paper 6: Agentic Information Retrieval

- **Authors**: Weinan Zhang, Junwei Liao, Ning Li, Kounianhua Du, Jianghao Lin
- **Year**: 2024
- **Source**: arXiv:2410.09713
- **Key Contribution**: Introduces "agentic IR" - a paradigm shift from static information retrieval to dynamic, context-dependent information states managed by LLM agents.
- **Methodology**: Conceptual framework with case studies
- **Key Concepts**:
  - Information state: user's current context including acquired items, preferences, and decision-making processes
  - Dynamic vs. static information retrieval
  - Task execution beyond just information filtering
- **Key Findings**:
  - Traditional IR limited by static corpus and passive retrieval
  - LLM agents can synthesize, manipulate, and generate new content
  - Agentic IR enables personalized, adaptive information delivery
- **Relevance to Our Research**: Framework for how AI research agents can dynamically present information based on user context rather than static output.

---

### Paper 7: FeedSum - Learning to Summarize from LLM-generated Feedback

- **Authors**: Hwanjun Song, Taewon Yun, Yuho Lee, Jihwan Oh, Gihun Lee, Jason Cai, Hang Su
- **Year**: 2024
- **Source**: arXiv:2410.13116
- **Key Contribution**: Introduces FeedSum dataset with multi-dimensional LLM feedback for training better summarizers. Shows smaller models can outperform larger ones with appropriate feedback training.
- **Methodology**: Created large-scale dataset of LLM feedback on summaries across three dimensions: faithfulness, completeness, conciseness. Compared supervised fine-tuning vs. direct preference optimization.
- **Datasets**: FeedSum (available on HuggingFace: DISLab/FeedSum)
- **Key Findings**:
  - High-quality, multi-dimensional, fine-grained feedback significantly improves summary generation
  - SummLlama3-8b outperforms Llama3-70b-instruct on human-preferred summaries
  - Three key dimensions: faithfulness, completeness, conciseness
- **Code Available**: Yes - HuggingFace (DISLab/SummLlama3-8B)
- **Relevance to Our Research**: Provides methodology and dataset for training summarizers that produce human-preferred output along specific quality dimensions.

---

### Paper 8: CARE - Collaborative AI Research Environment

- **Authors**: John P. Lalor, Xiaoyu (Rosie) Yang, Huasong Leng, Zikui Cai, et al.
- **Year**: 2024
- **Source**: arXiv:2410.24032
- **Key Contribution**: System for collaborative research between humans and AI, focusing on effective information presentation and research workflows.
- **Methodology**: System design and evaluation
- **Key Findings**:
  - Importance of structured information presentation in research contexts
  - Collaborative workflows between humans and AI improve research outcomes
  - Need for clear provenance and source attribution
- **Relevance to Our Research**: Model for how to present AI research outputs in ways that support human comprehension and collaboration.

---

### Paper 9: Reference-Free Evaluation Metrics for Summarization

- **Authors**: (From arXiv:2501.12011)
- **Year**: 2025
- **Key Contribution**: Methods for evaluating summary quality without reference summaries
- **Methodology**: Evaluation framework comparison
- **Key Findings**:
  - Reference-free metrics can assess summary quality based on source document alone
  - Enables evaluation in scenarios without gold-standard summaries
  - Important for real-world deployment where references unavailable
- **Relevance to Our Research**: Enables automated assessment of AI-to-human communication quality without human-written references.

---

### Paper 10: Cognitive Load in Streaming LLM Output

- **Authors**: (From arXiv:2504.17999)
- **Year**: 2025
- **Key Contribution**: Studies cognitive load when consuming streaming LLM output
- **Methodology**: User studies measuring cognitive load during interaction
- **Key Findings**:
  - Streaming output affects cognitive load differently than batch output
  - Pacing and chunking influence comprehension
  - Need to balance speed with cognitive processing capacity
- **Relevance to Our Research**: Critical for understanding how to pace and structure AI output delivery for human comprehension.

---

### Paper 11: Generative Interfaces for LLM Systems

- **Authors**: (From arXiv:2508.19227)
- **Year**: 2025
- **Key Contribution**: Design patterns for LLM interfaces that improve usability
- **Methodology**: Interface design study
- **Key Findings**:
  - Interface design significantly impacts user comprehension
  - Progressive revelation patterns effective for complex information
  - Visual hierarchy and formatting aid information processing
- **Relevance to Our Research**: Practical design patterns for presenting AI output effectively.

---

## Common Methodologies

### User Studies
- Used in Papers 1, 2, 8, 10, 11
- Measure comprehension, cognitive load, user satisfaction
- Often compare multiple interface conditions

### LLM-as-Evaluator
- Used in Papers 4, 7
- LLMs evaluate quality of generated content
- More consistent with human judgment than traditional metrics

### Progressive/Incremental Design
- Used in Papers 1, 5, 11
- Start with simplified information, reveal complexity progressively
- Helps users build mental models

---

## Standard Baselines

### Summarization Baselines
- BART, T5, Pegasus (encoder-decoder models)
- GPT-3.5/4 zero-shot and few-shot
- Fine-tuned domain-specific models

### Evaluation Baselines
- ROUGE (1, 2, L)
- BERTScore
- Human evaluation (faithfulness, coherence, relevance)

---

## Evaluation Metrics

### Automatic Metrics
- **ROUGE-1/2/L**: N-gram overlap with reference
- **BERTScore**: Semantic similarity using BERT embeddings
- **SummaC**: Consistency checking between summary and source
- **QuestEval**: Question-answering based evaluation

### Human Evaluation Dimensions
- **Faithfulness**: Summary consistent with source (no hallucination)
- **Completeness**: Key information included
- **Conciseness**: No unnecessary verbosity
- **Coherence**: Logical flow and readability
- **Relevance**: Appropriate for user needs

### Cognitive Metrics
- Task completion time
- Comprehension accuracy
- Self-reported cognitive load (NASA-TLX)
- Eye-tracking measures

---

## Datasets in the Literature

### Summarization Datasets
- **CNN/DailyMail**: News summarization benchmark
- **XSum**: Extreme summarization
- **FeedSum** (HuggingFace): Multi-dimensional feedback on summaries
- **Patent documents**: Domain-specific evaluation

### Human-AI Interaction Datasets
- Custom task-based studies common
- Information seeking tasks
- Decision-making scenarios

---

## Gaps and Opportunities

### Gap 1: Research-Specific Communication
Most work focuses on chat interfaces or general summarization. Little research specifically addresses communicating research findings (methods, results, uncertainties) from AI agents to human researchers.

### Gap 2: Trust and Verification
While summarization reduces information volume, it risks hiding important details. Need methods that maintain user trust and enable verification when needed.

### Gap 3: Adaptive Communication
Limited work on systems that adapt communication style based on user expertise, task context, or time constraints.

### Gap 4: Multi-Modal Research Output
Most work focuses on text. Research outputs may benefit from combined text, visualizations, and structured data.

---

## Recommendations for Our Experiment

### Recommended Approaches to Test
1. **Progressive Disclosure**: Present high-level summary first, allow drill-down to details
2. **Multi-Dimensional Summaries**: Separate findings by faithfulness/completeness/conciseness
3. **Structured Output Formats**: Use clear sections, headers, bullet points
4. **Cognitive Load Management**: Chunk information, use visual hierarchy
5. **Interactive Exploration**: Allow users to request more detail on specific aspects

### Recommended Datasets
1. **FeedSum** (HuggingFace: DISLab/FeedSum) - For training/testing summarization approaches
2. **Custom research output dataset** - Create sample AI research outputs to test communication methods

### Recommended Evaluation Metrics
1. **Human evaluation**: Comprehension accuracy, time to understanding, trust ratings
2. **LLM-as-judge**: Rate summary quality on key dimensions
3. **Task completion**: Can users make decisions based on the communicated information?
4. **Cognitive load measures**: Self-report or behavioral proxies

### Methodological Considerations
- Use within-subjects design to compare communication methods
- Include users with varying expertise levels
- Measure both immediate comprehension and retention
- Consider trade-offs: more detail increases accuracy but cognitive load

---

## References

1. Springer, A., & Whittaker, S. (2018). Progressive Disclosure: Designing for Effective Transparency. arXiv:1811.02164
2. Wang, B., Liu, J., Karimnazarov, J., & Thompson, N. (2024). Task Supportive and Personalized Human-Large Language Model Interaction. ACM CHIIR 2024.
3. Zhang, Y., et al. (2024). A Comprehensive Survey on Automatic Text Summarization with LLM Methods. arXiv:2403.02901
4. Nguyen, H., et al. (2024). A Comparative Study of Quality Evaluation Methods for Text Summarization. arXiv:2407.00747
5. Wasi, A.T., & Islam, M.R. (2024). CogErgLLM: Cognitive Ergonomics in LLM System Design. arXiv:2407.02885
6. Zhang, W., et al. (2024). Agentic Information Retrieval. arXiv:2410.09713
7. Song, H., et al. (2024). Learning to Summarize from LLM-generated Feedback. arXiv:2410.13116
8. Lalor, J.P., et al. (2024). CARE: Collaborative AI Research Environment. arXiv:2410.24032
9. Reference-Free Evaluation Metrics. (2025). arXiv:2501.12011
10. Cognitive Load in Streaming LLM. (2025). arXiv:2504.17999
11. Generative Interfaces for LLM. (2025). arXiv:2508.19227
