# 1. Model Architectures
Both models use identical hyperparameters and preprocessing for a fair comparison.

## LSTM Model:
Embedding(vocab_size, 64) → Dropout(0.3) → LSTM(128) → Dropout(0.3) → Dense(64,
ReLU) → Dense(1, Sigmoid)

## Bi-LSTM Model:
Embedding(vocab_size, 64) → Dropout(0.3) → Bidirectional(LSTM(128)) → Dropout(0.3)
→ Dense(64, ReLU) → Dense(1, Sigmoid)

## Shared Configuration:
    • Preprocessing: URL/HTML removal, user_mention removal, stopword removal, non- alphabet removal, lowercase tokenization.

    • Vocabulary size ~17,000 words; max sequence length   ~15 tokens.

    • Optimizer: Adam

    • Loss: Binary Crossentropy

    • Batch size: 64; Max epochs: 10. 

    Regularization: Dropout(0.3),EarlyStopping (patience=3, restore best weights), validation split 10%.
## 2. Performance Comparison




## 3. Discussion

# Why Bi-LSTM Doesn't Significantly Improve Performance
Despite processing sequences in both forward and backward directions, Bi-LSTM achieves
only a negligible +0.03% improvement over standard LSTM. Three factors explain this.
First, tweets are very short (~15 tokens after preprocessing). A single forward LSTM pass
already captures the full sequence context by the final token, leaving little room for backward
context to add new information. Second, this classification task is largely keyword-driven —
medical terms such as xanax, tylenol, pain, and fever are strong, position-independent
predictors of personal health mentions. The forward LSTM encodes these tokens regardless
of where they appear. Third, Bi-LSTM's 9% larger parameter count (1.30M vs 1.19M)
increases overfitting risk. As seen in the training curves, both models overfit — validation
loss rises while training loss falls — but Bi-LSTM's validation accuracy peaks earlier and
declines more sharply (83% → 80.5%), indicating the extra capacity memorizes noise rather
than learning generalizable patterns.

# Why Stacking LSTM/Bi-LSTM Layers Is Not Optimal
Stacking multiple recurrent layers is not beneficial for this task for three reasons. First,
stacked architectures are designed to learn hierarchical temporal patterns over long sequences
such as paragraphs or documents. Tweets at ~15 tokens are too short for multi-level
hierarchical modeling to be meaningful. Second, the current single-layer models already
exhibit overfitting. Stacking would roughly double or triple the parameter count (single
LSTM ~1.2M; two-layer LSTM ~1.8M), severely worsening overfitting against a training set
of only ~13,300 samples. Third, when token presence is highly predictive — as it is here with
medical keywords — deeper recurrent stacks add computational complexity but no additional
discriminative power, resulting in accuracy that plateaus or regresses.
Why This Architecture Is the Best Choice
The single-layer design is optimal for this task because it balances capacity and
generalization. A 64-dimensional embedding adequately represents a medical-domain
vocabulary. 128 LSTM units provide sufficient hidden-state depth to model short tweet
sequences without overfitting. The Dense(64, ReLU) intermediate layer compresses the
recurrent output into the most discriminative features before the sigmoid output. Dropout at
0.3 applied after both the embedding and LSTM layers, combined with EarlyStopping, keeps
generalization stable. Compared to Bi-LSTM and stacked alternatives, this architecture
achieves equivalent test performance with fewer parameters, faster training, and better
validation stability — making it the most practical and efficient model for this dataset and
task.
4. Conclusion
Both LSTM and Bi-LSTM achieve approximately 81.7% test accuracy. The 0.03% difference
in favour of Bi-LSTM is statistically insignificant. Bidirectional context and stacked layers
provide no meaningful benefit for short-text, keyword-driven classification with a moderately
sized dataset. The primary performance bottleneck is class imbalance (71% Non-Personal vs
29% Personal Health), which causes both models to underperform on minority-class recall
(~58–59%). The single-layer LSTM is the recommended model — it matches Bi-LSTM's
accuracy with fewer parameters and greater training stability.
