# Stock Prediction Analysis

This project implements a robust time-series cross-validation model to predict stock price movements for various stocks. The results are analyzed to identify the most predictable stocks and provide actionable insights for trading and portfolio construction.

---

## Key Findings

### Best Predictable Stocks (Balance of Accuracy and F1)

1. **BOP**:

   - Accuracy: **93.86%**
   - F1 Score: **0.967**
   - Interpretation: `BOP` is the most predictable ticker with both high accuracy and F1 score. It performs well in predicting both classes (up and down).

2. **SNBL**:

   - Accuracy: **84.56%**
   - F1 Score: **0.911**
   - Interpretation: `SNBL` is another highly predictable ticker with strong performance in both accuracy and F1 score.

3. **FCCL**:
   - Accuracy: **69.41%**
   - F1 Score: **0.799**
   - Interpretation: `FCCL` has moderate accuracy but a strong F1 score, indicating good performance in predicting both classes.

---

### Concerning Models (High Accuracy but Low F1)

1. **FFC**:

   - Accuracy: **63.49%**
   - F1 Score: **0.112**
   - Issue: Despite decent accuracy, the very low F1 score suggests poor performance in predicting the minority class. The model likely predicts one class most of the time.

2. **SYS**:

   - Accuracy: **61.73%**
   - F1 Score: **0.091**
   - Accuracy Std: **0.389**
   - Issue: `SYS` has inconsistent performance across folds (high standard deviation) and a very low F1 score, indicating poor prediction quality.

3. **POL**:
   - Accuracy: **58.44%**
   - F1 Score: **0.102**
   - Issue: Similar to `FFC` and `SYS`, `POL` has poor performance in predicting the minority class.

---

## How to Use This Data

### For Trading

- **Focus on BOP, SNBL, and FCCL** for potential trading strategies as they show the most promise.
- **Use as a directional signal**, not for timing: The model predicts direction (up/down) but not the magnitude of price changes.
- **Consider class imbalance**:
  - BOP: High F1 score and consistent predictions.
  - SNBL: Strong performance across both metrics.
  - FCCL: Balanced performance with a strong F1 score.

### Portfolio Construction

- **Weighted Allocation**:
  - Assign more capital to stocks with:
    - Higher accuracy (>70%)
    - Higher F1 scores (>0.80)
    - Lower standard deviation (<0.10)
- **Risk Management**:
  - Use stop losses regardless of predictions.
  - Consider position sizing based on model confidence.
  - Be cautious with stocks showing extreme class imbalance.

---

## Implementation Suggestions

1. **Start Small**: Test the predictions on a small portfolio first.
2. **Combine Signals**: Use these predictions alongside other analyses (e.g., fundamentals, technical indicators).
3. **Monitor Performance**: Track how well predictions match actual movements.
4. **Refine the Model**:
   - Add market sentiment features.
   - Incorporate macroeconomic indicators.
   - Explore cross-stock relationships.

---

## Limitations

- **Class Imbalance**: Many stocks show heavy bias toward one direction, which can skew predictions.
- **Accuracy Ceiling**: Even the best models typically achieve 55-65% accuracy in stock prediction.
- **Risk of Overfitting**: Models with high accuracy but low F1 scores may overfit to the majority class.

---

## Conclusion

This analysis highlights the most predictable stocks and provides actionable insights for trading and portfolio management. However, proper risk management and further refinement of the model are essential for practical implementation.
