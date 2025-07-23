# AI Narrative Index (AINI) – Variable Overview

This table describes the core variables created in the `final_df` DataFrame as part of the AI Narrative Index construction pipeline.

| Variable Name            | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `date`                   | The date of the article publication (daily resolution).                     |
| `normalized_AINI`        | Daily ratio of narrative articles to total articles (weighted AINI).        |
| `simple_AINI`            | Raw count of articles classified as narrative per day.                      |
| `MA_7`                   | 7-day simple moving average of `normalized_AINI`.                           |
| `MA_30`                  | 30-day simple moving average of `normalized_AINI`.                          |
| `EMA_02`                 | Exponential moving average of `normalized_AINI` with alpha = 0.2.           |
| `EMA_04`                 | Exponential moving average of `normalized_AINI` with alpha = 0.4.           |
| `EMA_06`                 | Exponential moving average of `normalized_AINI` with alpha = 0.6.           |
| `EMA_08`                 | Exponential moving average of `normalized_AINI` with alpha = 0.8.           |
| `normalized_AINI_growth`| First difference of `normalized_AINI`, capturing daily growth.              |
| `relative_AINI_weekly`  | Growth of `normalized_AINI` relative to its 7-day moving average.           |
| `relative_AINI_month`   | Growth of `normalized_AINI` relative to its 30-day moving average.          |