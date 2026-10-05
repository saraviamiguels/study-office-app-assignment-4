# 🎓 Week 6 at the study office

A Streamlit app for the study office (Business Data Science, Aalborg University, session 10 assignment).
About one first-year student in five leaves during the first semester. The office can hold about 40 conversations
at the end of week 6. This app shows **whom to offer a conversation** this week, ranked by a model's risk, and
**what mistakes** a rule makes: students reached in time, worried for nothing, or missed.

## What the app shows
1. **This week's list**: the 2026 students ranked by risk, with the top N marked and the reasons in plain words.
2. **The rule and its mistakes**: a slider (number of conversations or a cut-off) and the four boxes on the 2025 cohort.
3. **Per group**: the same numbers for international and domestic students.
4. **Costs** and **One student**: our own additions; the costs as inputs, and why a single student is on the list.

## Files
| File | What it is |
|---|---|
| `app.py` | the app |
| `portable.py` | the hotel app's loader (aaubs/tonights-front-desk), unchanged: reads the model without a pickle |
| `model/` | the XGBoost model from step 9 of the notebook: `preprocess.json` + `booster.json` |
| `data/` | the two CSV files (the app falls back to GitHub if they are missing) |

## Run it
Locally: `pip install -r requirements.txt`, then `streamlit run app.py`.
Online: on [share.streamlit.io](https://share.streamlit.io), *Create app* → this repository → `app.py`.

*The students are synthetic, resampled from UCI "Predict students' dropout and academic success" (Realinho et al.
2022, CC BY 4.0). The model only suggests; an adviser decides, and students are told and can object.*
