"""Week 6 at the study office: who should the advisers talk to, and what mistakes will that rule make?

Built on the structure of the hotel app (aaubs/tonights-front-desk). The model sits next to the code, in model/:
preprocess.json (filling gaps, scaling and one-hot, as plain numbers) + booster.json (the XGBoost trees), both
written in step 9 of the notebook with the hotel app's portable.py. The data is read from data/ if it is there,
otherwise straight from GitHub, like the notebook.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from portable import Model

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
DATA_URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"
NAVY, LAV, GREY, ORANGE = "#211A52", "#C2C1CC", "#54616E", "#D2542A"
CAPACITY = 40                                  # conversations three advisers can hold at the end of week 6
COSTS = {"conversation": 500, "false_alarm": 2_000, "dropout": 60_000, "share_helped": 0.30}   # step 6 of the notebook

# plain words for the reasons, and which kind of conversation each reason calls for
WHY = {"age": lambda v: f"age {v:.0f}", "admission_grade": lambda v: f"admission grade {v:.1f}",
       "international": lambda v: "international student" if v == 1 else "domestic student",
       "first_gen": lambda v: "first in family at university" if v == 1 else "parents went to university",
       "su_scholarship": lambda v: "has SU grant" if v == 1 else "no SU grant",
       "fees_owed": lambda v: "owes fees" if v == 1 else "no fees owed",
       "moved_from_home": lambda v: "moved from home" if v == 1 else "lives at home",
       "married": lambda v: "married" if v == 1 else "not married",
       "evening_programme": lambda v: "evening programme" if v == 1 else "day programme",
       "programme": lambda v: f"studies {v}", "gender": lambda v: f"{v}",
       "logins_total": lambda v: f"{v:.0f} logins in weeks 1-6", "logins_last3": lambda v: f"{v:.0f} logins in the last 3 weeks",
       "logins_trend": lambda v: f"logins trend {v:+.1f}/week", "submitted_share": lambda v: f"handed in {v:.0%} of assignments",
       "missed_last3": lambda v: f"missed {v:.0f} of the last 3 assignments",
       "quiz_mean": lambda v: "no quiz score" if pd.isna(v) else f"quiz average {v:.0f}",
       "weeks_since_login": lambda v: f"{v:.0f} weeks since last login"}
THEME = {"fees_owed": "money", "su_scholarship": "money",
         "logins_total": "studies", "logins_last3": "studies", "logins_trend": "studies", "submitted_share": "studies",
         "missed_last3": "studies", "weeks_since_login": "studies", "quiz_mean": "studies"}   # everything else: background
TALK = {"studies": "📚 studies: planning, a study group, a tutor",
        "money": "💰 money: SU, fees, a payment plan",
        "background": "🧭 situation: work, family, the right programme? (check first)",
        "-": "-"}

st.set_page_config(page_title="Week 6 at the study office", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------- the model and the data, loaded once
@st.cache_resource
def load_model():
    return Model(MODEL_DIR)                    # no pickle: loads with any recent pandas and xgboost


@st.cache_data
def load_data():
    read = lambda f: pd.read_csv(HERE / "data" / f) if (HERE / "data" / f).exists() else pd.read_csv(DATA_URL + f)
    history, new = read("history_week6.csv"), read("new_week6.csv")
    model = load_model()
    last = history[history["cohort"] == history["cohort"].max()].copy()    # 2025: the cohort the model never saw
    last["risk"] = model.predict_proba(last)
    new["risk"] = model.predict_proba(new)
    new["rank"] = new["risk"].rank(ascending=False, method="first").astype(int)
    return history, last, new.sort_values("rank")


def explain(model, rows, top=3):
    """The top reasons in words, and the theme with the biggest push (one-hot parts summed back per feature)."""
    contrib = model.contributions(rows)
    out = []
    for idx, row in contrib.iterrows():
        up = row[row > 0].sort_values(ascending=False)
        themes = up.groupby(lambda f: THEME.get(f, "background")).sum()
        out.append({"why": " · ".join(WHY[f](rows.at[idx, f]) for f in up.index[:top]),
                    "theme": themes.idxmax() if len(themes) else "-"})
    return pd.DataFrame(out, index=rows.index)


def apply_rule(data, rule):
    """True for every student the rule says to contact: the n highest risks, or everyone above the cut-off."""
    if rule["kind"] == "n":
        return data["risk"].rank(ascending=False, method="first") <= rule["n"]
    return data["risk"] >= rule["cut"]


def boxes(data, flag):
    left = data["left"] == 1
    TP, FP = int((flag & left).sum()), int((flag & ~left).sum())
    FN, TN = int((~flag & left).sum()), int((~flag & ~left).sum())
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": TP / max(1, TP + FP), "recall": TP / max(1, TP + FN)}


def net_value(b, costs):
    return b["TP"] * costs["helps"] * costs["leave"] - (b["TP"] + b["FP"]) * costs["talk"] - b["FP"] * costs["worry"]


model = load_model()
history, last, new = load_data()
YEAR = int(last["cohort"].iloc[0])

# ---------------------------------------------------------------- sidebar: the rule
with st.sidebar:
    st.header("📏 The rule")
    kind = st.radio("How does the office choose?", ["Talk to the highest risks", "Contact everyone above a cut-off"],
                    help="With a fixed number of conversations, the rule is 'the N highest risks'.")
    if kind.startswith("Talk"):
        n = st.slider("Number of conversations", 10, 200, CAPACITY, 1, key="n_talks",
                      help="Three advisers can hold about 40. A fourth adviser adds about 13.")
        rule = {"kind": "n", "n": n}
        st.caption(f"Talk to the **{n}** students with the highest risk.")
    else:
        cut = st.slider("Cut-off (risk)", 0.02, 0.80, 0.30, 0.01, format="%.2f", key="cut",
                        help="Contact every student whose risk is at least this high.")
        rule = {"kind": "cut", "cut": cut}
        st.caption(f"This week that means **{int((new['risk'] >= cut).sum())}** conversations.")
    st.divider()
    st.caption(f"Model: XGBoost (`model/booster.json`) · learned from 2023–2024 · checked on {YEAR}")

st.title("🎓 Week 6 at the study office")
st.markdown("Which first-year students should the advisers **offer** a conversation this week, and what mistakes "
            "does that rule make? *The model ranks, an adviser decides, and students can always say no.*")
tabs = st.tabs(["📋 This week's list", "⚖️ The rule and its mistakes", "🌍 Per group", "💶 Costs",
                "🔎 One student", "⚙️ Under the hood"])

# ---------------------------------------------------------------- 1 · this week's list
with tabs[0]:
    new["on the list"] = apply_rule(new, rule)
    listed = new[new["on the list"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students this week", f"{len(new):,}")
    c2.metric("On the list", len(listed))
    c3.metric("Expected to leave, among them", f"≈ {listed['risk'].sum():.0f}",
              help="The sum of their risks: if the risks are right, about this many would leave without help.")
    c4.metric("International on the list", f"{int(listed['international'].sum())} of {len(listed)}",
              help=f"International students are {new['international'].mean():.0%} of this year's cohort.")

    f1, f2 = st.columns([2, 1])
    programmes = f1.multiselect("Programme", sorted(new["programme"].unique()), placeholder="All programmes", key="prog")
    only = f2.toggle("Only students on the list", value=True, key="only")
    view = new[new["on the list"]] if only else new
    if programmes:
        view = view[view["programme"].isin(programmes)]
    view = view.head(300).copy()               # the reasons are computed for at most 300 rows
    if view.empty:
        st.warning("No students match these filters. Choose another programme, or switch off "
                   "'Only students on the list'.")
    else:
        view = view.join(explain(model, view))
        view["conversation"] = view["theme"].map(TALK)
        view["international"] = view["international"].map({0: "no", 1: "yes"})
        view["risk %"] = (100 * view["risk"]).round(0)
        show = ["rank", "student_id", "programme", "international", "risk %", "on the list", "why", "conversation"]
        st.dataframe(view[show], hide_index=True, width="stretch", height=520,
                     column_config={"risk %": st.column_config.ProgressColumn("risk", min_value=0, max_value=100, format="%d%%"),
                                    "on the list": st.column_config.CheckboxColumn("on the list"),
                                    "why": st.column_config.TextColumn("why (biggest reasons)", width="large")})
        st.download_button("⬇️ Download this list (CSV)", view[show].to_csv(index=False), "week6_list.csv", "text/csv")
    st.info("**How to use this list.** It *suggests* whom to offer a conversation, with the reasons. The adviser "
            "decides: skip students already in contact or who asked not to be contacted, check students who are "
            "on the list mainly because of their background, and never tell a student 'the model says you will drop "
            "out'. Students are told this system exists and can object.")

# ---------------------------------------------------------------- 2 · the mistakes of the rule, on last year's cohort
with tabs[1]:
    flag = apply_rule(last, rule)
    b = boxes(last, flag)
    st.markdown(f"**What if the office had used this rule last year?** For the **{len(last)} students of {YEAR}** we "
                f"know what happened: **{int(last['left'].sum())} left** later in the semester.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Reached in time", b["TP"], help="Contacted, and really at risk: they left later (true positives).")
    c2.metric("📞 Worried for nothing", b["FP"], help="Contacted, but they would have stayed anyway (false positives).")
    c3.metric("🚪 Missed", b["FN"], help="Not contacted, and they left (false negatives).")
    c4.metric("✅ Fine, left alone", b["TN"], help="Not contacted, and they stayed (true negatives).")
    st.success(f"In plain words: the office talks to **{b['TP'] + b['FP']} students**. **{b['TP']}** of them are "
               f"reached in time and **{b['FP']}** are worried for nothing. **{b['FN']}** students who leave are "
               f"never contacted.")
    p1, p2, p3 = st.columns(3)
    p1.metric("Precision", f"{b['precision']:.0%}", help="Of the students we talked to, the share who really left.")
    p1.caption(f"Of the {b['TP'] + b['FP']} students we talked to, {b['precision']:.0%} really were at risk.")
    p2.metric("Recall", f"{b['recall']:.0%}", help="Of the students who left, the share we reached.")
    p2.caption(f"Of the {b['TP'] + b['FN']} students who left, we reached {b['recall']:.0%}.")
    random_tp = (b["TP"] + b["FP"]) * last["left"].mean()
    p3.metric("Picked at random instead", f"≈ {random_tp:.0f} reached",
              help="The same number of students picked at random would contain about this many leavers.")
    p3.caption(f"The model finds about {b['TP'] / max(random_tp, 0.1):.0f}× more than chance.")

    # the trade-off for every number of conversations, with the current rule marked
    order = last.sort_values("risk", ascending=False)["left"].to_numpy()
    k = np.arange(1, 301)
    tp = order.cumsum()[:300]
    curve = pd.DataFrame({"conversations": np.concatenate([k, k, k]),
                          "students": np.concatenate([tp, k - tp, last["left"].sum() - tp]),
                          "box": ["reached in time"] * 300 + ["worried for nothing"] * 300 + ["missed"] * 300})
    lines = alt.Chart(curve).mark_line(strokeWidth=2.5).encode(
        x=alt.X("conversations", title="number of conversations (highest risks first)"),
        y=alt.Y("students", title=f"students ({YEAR})"),
        color=alt.Color("box", scale=alt.Scale(domain=["reached in time", "worried for nothing", "missed"],
                                               range=[ORANGE, LAV, NAVY]), legend=alt.Legend(orient="top", title=None)))
    rule_line = alt.Chart(pd.DataFrame({"x": [b["TP"] + b["FP"]]})).mark_rule(color=GREY, strokeDash=[4, 4]).encode(x="x")
    st.altair_chart((lines + rule_line).properties(height=320), width="stretch")
    st.caption("No rule removes both mistakes: more conversations reach more students who leave, but worry more "
               "students who were fine. The dashed line is the current rule.")

# ---------------------------------------------------------------- 3 · per group
with tabs[2]:
    flag = apply_rule(last, rule)
    rows = {}
    for name, g in [("domestic", last["international"] == 0), ("international", last["international"] == 1)]:
        bg = boxes(last[g], flag[g])
        rows[name] = {"students": int(g.sum()), "really left": 100 * last.loc[g, "left"].mean(),
                      "average predicted risk": 100 * last.loc[g, "risk"].mean(),
                      "talked to": bg["TP"] + bg["FP"], "reached in time": bg["TP"], "worried for nothing": bg["FP"],
                      "missed": bg["FN"], "recall": 100 * bg["recall"], "precision": 100 * bg["precision"]}
    st.markdown(f"The same rule, on the {YEAR} cohort, for **domestic** and **international** students.")
    pct = lambda label: st.column_config.NumberColumn(label, format="%.1f%%")
    st.dataframe(pd.DataFrame(rows).T, width="stretch",
                 column_config={c: pct(c) for c in ["really left", "average predicted risk", "recall", "precision"]})
    c1, c2 = st.columns(2)
    for col, name in zip([c1, c2], rows):
        r = rows[name]
        col.markdown(f"**{name.capitalize()}:** of the {r['reached in time'] + r['missed']} {name} students who left, "
                     f"**{r['reached in time']} were reached in time** and **{r['missed']} were missed** "
                     f"(recall {r['recall']:.0f} %). {r['worried for nothing']} {name} students were worried for nothing.")
    logins = history.groupby(["international", "left"])["logins_last3"].mean()
    st.warning(f"**Read with care.** Both groups leave about equally often, but **international students log in much "
               f"less, whether they stay or leave**: an international student who stays logs in "
               f"{logins[(1, 0)]:.1f} times in the last three weeks, a domestic student who leaves {logins[(0, 1)]:.1f} "
               f"times. A login is not the same signal for everyone. The groups are small "
               f"({int(last.loc[last['international'] == 1, 'left'].sum())} international students left in {YEAR}), "
               f"so one student more or less changes these percentages a lot. Which fairness goal to aim for "
               f"(equal recall, or risks that mean the same in both groups) is a decision for people.")

# ---------------------------------------------------------------- 4 · costs (our own addition)
with tabs[3]:
    st.markdown("**What is a conversation worth?** These numbers are *assumptions*, a policy choice. Change them and "
                "see how the best rule moves.")
    c = COSTS
    i1, i2, i3, i4 = st.columns(4)
    costs = {"talk": i1.number_input("A conversation (DKK)", 0, 5_000, c["conversation"], 100),
             "worry": i2.number_input("A false alarm, for the student (DKK)", 0, 50_000, c["false_alarm"], 500),
             "leave": i3.number_input("A student who leaves (DKK)", 0, 300_000, c["dropout"], 5_000),
             "helps": i4.slider("Share of students at risk a conversation keeps", 0.0, 1.0, c["share_helped"], 0.05)}
    gain = costs["helps"] * costs["leave"]
    breakeven = (costs["talk"] + costs["worry"]) / (gain + costs["worry"]) if gain + costs["worry"] > 0 else 1.0
    order = last.sort_values("risk", ascending=False)["left"].to_numpy()
    ks = np.arange(0, len(order) + 1)
    tp = np.concatenate([[0], order.cumsum()])
    values = tp * gain - ks * costs["talk"] - (ks - tp) * costs["worry"]
    k_best = int(np.argmax(values))
    current = int(apply_rule(last, rule).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Break-even risk", f"{breakeven:.1%}", help="Contact a student when the risk is above this.")
    m2.metric(f"Best number of conversations ({YEAR})", k_best, help="The number of top-risk students that gave the "
              f"highest net value on the {YEAR} cohort.")
    m3.metric("Net value of the current rule", f"{values[current]:,.0f} DKK",
              f"{values[current] - values[k_best]:+,.0f} vs the best", delta_color="normal")
    chart = pd.DataFrame({"conversations": ks[:401], "net value (DKK)": values[:401]})
    line = alt.Chart(chart).mark_line(color=NAVY, strokeWidth=2.5).encode(x="conversations", y="net value (DKK)")
    marks = alt.Chart(pd.DataFrame({"x": [current, k_best], "what": ["current rule", "best"]})).mark_rule(strokeWidth=2).encode(
        x="x", color=alt.Color("what", scale=alt.Scale(domain=["current rule", "best"], range=[GREY, ORANGE]),
                               legend=alt.Legend(orient="top", title=None)))
    st.altair_chart((line + marks).properties(height=300), width="stretch")
    extra = values[min(current + 13, len(values) - 1)] - values[current]
    st.markdown(f"**A fourth adviser** (about 13 more conversations) would change the net value by **{extra:+,.0f} DKK** "
                f"under these assumptions, before the adviser's salary. "
                + ("More capacity would pay off." if extra > 0 else "More capacity would not pay off."))
    st.caption("Net value = reached in time × share kept × cost of leaving − conversations × cost of a conversation "
               "− false alarms × cost of a false alarm. The weakest assumption is the share a conversation keeps: "
               "it should be measured, not guessed.")

# ---------------------------------------------------------------- 5 · one student (our own addition)
with tabs[4]:
    risk_of = new.set_index("student_id")["risk"]
    pick = st.selectbox("Student (highest risk first)", new["student_id"], key="student",
                        format_func=lambda s: f"{s} · risk {risk_of[s]:.0%}")
    one = new[new["student_id"] == pick]
    r = one.iloc[0]
    left_col, right_col = st.columns([1, 2])
    left_col.metric("Risk of leaving", f"{r['risk']:.0%}")
    left_col.progress(float(r["risk"]))
    left_col.markdown(f"Rank **{r['rank']}** of {len(new)} · {'**on the list**' if r['on the list'] else 'not on the list'}")
    why = explain(model, one).iloc[0]
    left_col.markdown(f"**Biggest reasons:** {why['why'] or 'nothing pushes the risk up'}")
    left_col.markdown(f"**Suggested conversation:** {TALK[why['theme']]}")
    contrib = model.contributions(one).iloc[0]
    parts = pd.DataFrame({"feature": [WHY[f](r[f]) for f in contrib.index], "push": contrib.values})
    parts = parts.reindex(parts["push"].abs().sort_values(ascending=False).index).head(10)
    bars = alt.Chart(parts).mark_bar().encode(
        x=alt.X("push", title="← lowers the risk · raises the risk →"), y=alt.Y("feature", sort=None, title=None),
        color=alt.condition("datum.push > 0", alt.value(ORANGE), alt.value(NAVY)))
    right_col.altair_chart(bars.properties(height=320, title="What moves this student's risk (vs. an average student)"),
                           width="stretch")
    with st.expander("Everything the office knows about this student at week 6"):
        st.dataframe(one[model.features].T.astype(str)
                     .rename(columns=lambda _: "value"), width="stretch")

# ---------------------------------------------------------------- 6 · under the hood
with tabs[5]:
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(f"""
### The model
- **What it predicts:** the probability that a first-year student who is still enrolled at the end of week 6
  leaves later in the semester.
- **What it uses:** only what the office knows at week 6: enrolment data, money and life situation, and six weeks
  of activity on the learning platform. ECTS points, the deregistration form and the last login of the semester
  were left out: they are only known later (leakage).
- **How:** a pipeline (fill gaps, scale numbers, one-hot the categories) + XGBoost, stored as `preprocess.json` +
  `booster.json` and loaded with the hotel app's `portable.py`.
- **Learned from** the 2023 and 2024 cohorts, **checked on** {YEAR}: of the 40 highest risks, 14 really left
  (precision 35 %), and they include 22 % of all students who left (recall).
- **Limits:** synthetic students; few students leave (65 in {YEAR}), so every number is uncertain. International
  students log in less whether they stay or leave: check the mistakes per group every year.
""")
    with right_col:
        st.markdown("### Before a university uses this")
        st.markdown("""
- **A person decides.** The list is a suggestion; an adviser chooses whom to contact.
- **Students are told** that the system exists, which data it uses and why, and **can object**.
- The conversation is an **offer of help**, never a sanction or a mark in a file.
- The **costs** and the **fairness goal** are set by the university with students, not by the model.
- Mistakes are **checked per group every year**, and the model is retrained on new cohorts.
- In the EU AI Act, AI that evaluates or steers students is **high-risk**: documentation, human oversight,
  transparency and monitoring are required from 2 December 2027.
""")
