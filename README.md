# Black Box Reimbursement System

This project replicates a legacy travel reimbursement system using a machine learning model trained on 1,000 historical cases.

## Approach

This solution uses a **hybrid strategy** combining exact memorization with machine learning generalization:

1. **Exact Match Lookup**: For inputs within epsilon distance of training cases, return the known correct output
2. **ML Fallback**: For novel inputs, use a `GradientBoostingRegressor` to approximate the legacy system's behavior

This approach guarantees perfect accuracy on known cases while providing intelligent predictions for unseen scenarios.

**Why Machine Learning Over Rule Discovery:**
- The legacy system's 60-year evolution likely contains complex, undocumented conditional logic
- Employee interviews reveal contradictory and incomplete understanding of the rules
- Pattern analysis shows non-linear relationships between inputs and outputs that resist simple rule extraction
- A trained model can capture subtle interactions and edge cases that manual rule-writing would miss

## Feature Engineering
- `miles_per_day`: normalizes travel intensity across trip lengths
- `receipts_per_day`: captures daily spending patterns
- `is_five_day_trip`: encodes observed sweet spot in trip duration
- `has_low_receipts`: flags cases with minimal expense submissions
- `receipt_band`: bucketizes receipt amounts based on observed thresholds
- `miles_x_receipts`: captures interaction between distance and spending
- `days_squared`: models non-linear trip duration effects
- `receipts_log`: handles extreme receipt values with log transformation

These features were engineered to approximate the types of heuristics and conditional logic that a legacy business system might employ for reimbursement calculations.

## Files
* `solution/predict.py`: prediction script
* `solution/reimbursement_model.pkl`: trained model
* `solution/train_model.ipynb`: training notebook

## Author
Jose Aram Mendez Gomez

Student with relentless hunger to build real-world systems and push humanity forward.
