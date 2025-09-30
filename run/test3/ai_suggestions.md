        # AI Suggestions (offline fallback)
        _No OPENAI_API_KEY detected; returning local heuristic guidance._

        ## Quick Risks
        - `Air Conditioning` missing 100.00%
- `Year Built` missing 100.00%
- `Area` missing 100.00%
- `Full Bathrooms` missing 100.00%
- `Half Bathrooms` missing 100.00%

        ## Cleaning Priorities
        1. Drop columns with missing rate ≥ threshold (if business-irrelevant).
        2. Deduplicate rows.
        3. Type coercion for object→numeric/datetime when safe.
        4. Impute numeric (median) and categorical (__MISSING__).
        5. Review high-missing columns; consider flags or removal.

        ## Modeling Readiness
        - Check target leakage, time order, and outliers.
        - Standardize categories; limit high-cardinality.

        ## Nice-to-have
        - Add *_was_na flags; plan outlier handling (IQR/winsorize).
