import streamlit as st
import joblib
import json
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model  = joblib.load(os.path.join(BASE_DIR, 'model_v2 (1).pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler_v2 (1).pkl'))
with open(os.path.join(BASE_DIR, 'benchmarks (1).json'), 'r') as f:
    bench = json.load(f)

st.set_page_config(page_title="Workplace Decision Risk Assessment", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%);
    background-attachment: fixed;
}
h1, h2, h3 {
    color: #C9A84C !important;
    letter-spacing: 0.5px;
}
.stExpander {
    border: 1px solid #C9A84C40 !important;
    border-radius: 8px !important;
    background: #1A1A2E !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #C9A84C, #A67C3A) !important;
    color: #0F0F1A !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    border-radius: 6px !important;
}
.stButton > button {
    border: 1px solid #C9A84C !important;
    color: #C9A84C !important;
    background: transparent !important;
    border-radius: 6px !important;
}
.stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, #C9A84C, #A67C3A) !important;
    color: #0F0F1A !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    border-radius: 6px !important;
    width: 100% !important;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #C9A84C, #E8C97A) !important;
}
[data-testid="metric-container"] {
    background: #1A1A2E !important;
    border: 1px solid #C9A84C40 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
.stRadio > label {
    color: #C9A84C !important;
    font-weight: 500 !important;
}
hr {
    border-color: #C9A84C40 !important;
}
.stAlert {
    border-radius: 8px !important;
    border-left: 3px solid #C9A84C !important;
}
.stSlider > div > div > div {
    background: #C9A84C !important;
}
.stCaption {
    color: #8888AA !important;
}
.stSelectbox > div > div {
    background: #1A1A2E !important;
    border: 1px solid #C9A84C40 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'data' not in st.session_state:
    st.session_state.data = {}

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

def z_score(val, mean, std):
    return (val - mean) / std if std > 0 else 0

# ── Progress bar ───────────────────────────────────────────
pages = ['Welcome', 'Demographics', 'Impulsivity', 'Stress',
         'Overconfidence', 'Scenarios', 'Results']
if st.session_state.page < len(pages):
    st.progress(st.session_state.page / (len(pages) - 1))
    st.caption(f"Step {st.session_state.page + 1} of {len(pages)}: {pages[st.session_state.page]}")

# ══════════════════════════════════════════════════════════
# PAGE 0 — WELCOME
# ══════════════════════════════════════════════════════════
if st.session_state.page == 0:
    st.title("Workplace Decision Risk Assessment")
    st.write("""
    This tool assesses your psychological profile and predicts your
    decision-making risk pattern in workplace scenarios.

    **What you will do:**
    - Answer questions about your general tendencies (5 min)
    - Complete a short knowledge task (3 min)
    - Respond to 6 workplace situations (5 min)

    **Total time: approximately 12–15 minutes**

    Your responses are not stored. This is a research and educational tool.
    """)
    st.info("There are no right or wrong answers. Please respond honestly.")
    if st.button("Begin Assessment", type="primary"):
        next_page()

# ══════════════════════════════════════════════════════════
# PAGE 1 — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════
elif st.session_state.page == 1:
    st.header("Background Information")

    age = st.number_input("Age", min_value=18, max_value=65, value=22)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
    job_type = st.selectbox("Which job environment feels more familiar to you?", [
        "Clerical / Administrative / Office",
        "Manufacturing / Production / Factory"
    ])
    experience = st.selectbox("Work experience", [
        "No work experience",
        "Less than 1 year",
        "1 to 3 years",
        "More than 3 years"
    ])

    col1, col2 = st.columns(2)
    with col2:
        if st.button("Next →", type="primary"):
            st.session_state.data['age'] = age
            st.session_state.data['gender'] = gender
            st.session_state.data['job_type'] = 0 if "Clerical" in job_type else 1
            st.session_state.data['experience'] = experience
            next_page()

# ══════════════════════════════════════════════════════════
# PAGE 2 — BIS-11 (IMPULSIVITY)
# Back button sits outside the form; form holds all questions + Next
# ══════════════════════════════════════════════════════════
elif st.session_state.page == 2:
    st.header("Part A: General Tendencies")
    st.write("How often does each statement apply to you?")
    st.caption("1 = Rarely/Never   2 = Occasionally   3 = Often   4 = Almost Always")

    if st.button("← Back"):
        prev_page()

    bis_items = [
        "I plan tasks carefully.",
        "I do things without thinking.",
        "I make up my mind quickly.",
        "I am happy-go-lucky.",
        "I don't pay attention.",
        "I have racing thoughts.",
        "I plan trips well ahead of time.",
        "I am self-controlled.",
        "I concentrate easily.",
        "I save regularly.",
        "I squirm at plays or lectures.",
        "I am a careful thinker.",
        "I plan for job security.",
        "I say things without thinking.",
        "I like to think about complex problems.",
        "I change jobs.",
        "I act on impulse.",
        "I get easily bored when solving thought problems.",
        "I act on the spur of the moment.",
        "I am a steady thinker.",
        "I change residences.",
        "I buy things on impulse.",
        "I can only think about one problem at a time.",
        "I change hobbies.",
        "I spend or charge more than I earn.",
        "I often have extraneous thoughts when thinking.",
        "I am more interested in the present than the future.",
        "I am restless at the theater or lectures.",
        "I like puzzles.",
        "I am future oriented."
    ]
    reverse_items = [0, 6, 7, 8, 9, 11, 12, 14, 19, 28, 29]

    with st.form(key="page_2_form"):
        bis_responses = []
        for i, item in enumerate(bis_items):
            r = st.radio(f"{i+1}. {item}", [1, 2, 3, 4],
                         horizontal=True, key=f"bis_{i}",
                         format_func=lambda x: str(x))
            bis_responses.append(r)

        submitted = st.form_submit_button("Next →", type="primary")
        if submitted:
            scores = list(bis_responses)
            for i in reverse_items:
                scores[i] = 5 - scores[i]
            st.session_state.data['impulsivity'] = sum(scores)
            next_page()
            st.rerun()

# ══════════════════════════════════════════════════════════
# PAGE 3 — PSS-10 (STRESS)
# ══════════════════════════════════════════════════════════
elif st.session_state.page == 3:
    st.header("Part B: Recent Experiences")
    st.write("In the last month, how often have you felt this way?")
    st.caption("0 = Never   1 = Almost Never   2 = Sometimes   3 = Fairly Often   4 = Very Often")

    if st.button("← Back"):
        prev_page()

    pss_items = [
        "Been upset because of something that happened unexpectedly?",
        "Felt unable to control the important things in your life?",
        "Felt nervous and stressed?",
        "Felt confident about your ability to handle your personal problems?",
        "Felt that things were going your way?",
        "Found that you could not cope with all the things you had to do?",
        "Been able to control irritations in your life?",
        "Felt that you were on top of things?",
        "Been angered because of things outside your control?",
        "Felt difficulties were piling up so high you could not overcome them?"
    ]
    reverse_pss = [3, 4, 6, 7]

    with st.form(key="page_3_form"):
        pss_responses = []
        for i, item in enumerate(pss_items):
            r = st.radio(f"{i+1}. {item}", [0, 1, 2, 3, 4],
                         horizontal=True, key=f"pss_{i}",
                         format_func=lambda x: str(x))
            pss_responses.append(r)

        submitted = st.form_submit_button("Next →", type="primary")
        if submitted:
            scores = list(pss_responses)
            for i in reverse_pss:
                scores[i] = 4 - scores[i]
            st.session_state.data['stress'] = sum(scores)
            next_page()
            st.rerun()

# ══════════════════════════════════════════════════════════
# PAGE 4 — OVERCONFIDENCE (KNOWLEDGE CHECK)
# ══════════════════════════════════════════════════════════
elif st.session_state.page == 4:
    st.header("Part C: Knowledge Check")
    st.write("Answer each question, then rate your confidence.")

    if st.button("← Back"):
        prev_page()

    trivia = [
        ("What is the capital city of Australia?",
         ["Sydney", "Melbourne", "Canberra", "Brisbane"], "Canberra"),
        ("How many bones are in the adult human body?",
         ["156", "206", "312", "98"], "206"),
        ("Which planet is closest in size to Earth?",
         ["Mars", "Venus", "Mercury", "Neptune"], "Venus"),
        ("In which year did World War II end?",
         ["1943", "1944", "1945", "1946"], "1945"),
        ("What is the chemical symbol for gold?",
         ["Go", "Gd", "Au", "Ag"], "Au"),
        ("Which organ produces insulin?",
         ["Liver", "Kidney", "Pancreas", "Stomach"], "Pancreas"),
        ("How many sides does a regular hexagon have?",
         ["5", "6", "7", "8"], "6"),
        ("Who wrote Romeo and Juliet?",
         ["Charles Dickens", "William Shakespeare", "Jane Austen", "Oscar Wilde"],
         "William Shakespeare"),
        ("Approximate speed of light in km/s?",
         ["30,000", "300,000", "3,000,000", "30,000,000"], "300,000"),
        ("Which country has the largest land area?",
         ["China", "United States", "Canada", "Russia"], "Russia"),
    ]

    with st.form(key="page_4_form"):
        answers, confidences = [], []
        for i, (q, opts, _) in enumerate(trivia):
            st.write(f"**Q{i+1}. {q}**")
            ans = st.radio("", opts, horizontal=True, key=f"tri_{i}",
                           label_visibility="collapsed")
            conf = st.select_slider(
                f"Confidence in Q{i+1}:",
                options=[50, 60, 70, 80, 90, 100], value=70, key=f"conf_{i}")
            answers.append(ans)
            confidences.append(conf)
            st.divider()

        submitted = st.form_submit_button("Next →", type="primary")
        if submitted:
            correct = [1 if answers[i] == trivia[i][2] else 0 for i in range(10)]
            accuracy = sum(correct) * 10
            mean_conf = np.mean(confidences)
            st.session_state.data['overconfidence'] = mean_conf - accuracy
            st.session_state.data['accuracy'] = accuracy
            st.session_state.data['mean_confidence'] = mean_conf
            next_page()
            st.rerun()

# ══════════════════════════════════════════════════════════
# PAGE 5 — SCENARIOS
# ══════════════════════════════════════════════════════════
elif st.session_state.page == 5:
    st.header("Part D: Workplace Situations")
    st.info("Select what you would genuinely do. There are no right or wrong answers.")

    if st.button("← Back"):
        prev_page()

    rubric = {
        'M1': [0, 1, 2, 2, 1],
        'M2': [0, 2, 0, 3, 1],
        'M3': [0, 2, 3, 0, 3],
        'C1': [0, 2, 1, 1, 0],
        'C2': [0, 3, 1, 2, 0],
        'C3': [0, 3, 0, 0, 2],
    }

    scenarios = [
        ("M1", "Manufacturing",
         "You are mid-shift on the stamping line. Your supervisor called you over for a two-minute conversation about this afternoon's batch target. You return and resume. Three minutes later a colleague shouts: Did you do your guard check before restarting? You almost certainly did it but have no clear memory. Stopping resets a safety interlock taking 7 minutes. The batch is 25 minutes behind and your supervisor is watching.",
         "Time left: 35 minutes. Interlock reset: 7 minutes.",
         ["Stop the machine, complete the full interlock reset and redo the guard check from scratch.",
          "Stop briefly, visually inspect the guard without triggering the full reset, then restart.",
          "Continue running and ask your colleague to confirm the guard position while the machine operates.",
          "Continue running. You do this automatically every shift and not remembering is normal.",
          "Flag to your supervisor right now so the decision is documented, then continue based on their instruction."]),
        ("M2", "Manufacturing",
         "You are running a packaging machine when the amber pressure warning activates. This exact warning appeared three times in six weeks, each time a faulty sensor. Today it appears again but you notice a faint acrid smell that fades within fifteen seconds. The maintenance technician is on break and not answering. Your team lead is at the other end of the floor.",
         "Maintenance response time: 20–25 minutes. Remaining production window: 40 minutes.",
         ["Stop immediately and escalate to the team lead in person, reporting both the warning and the smell.",
          "Continue running, monitor for ten minutes, escalate only if smell reappears or second warning activates.",
          "Stop the machine, log both cues in the shift report, wait for maintenance even if it means missing the target.",
          "Continue running. The warning has always been a sensor fault and an uncertain smell is not unusual enough to justify shutdown.",
          "Try maintenance once more, if unreachable continue running while messaging the team lead."]),
        ("M3", "Manufacturing",
         "Policy requires cut-resistant gloves for all blade inspection tasks. The glove station is 45 seconds away. You need to do a pre-startup blade alignment check under a minute. Your team lead said the shift is behind and every station needs to stay on pace. Three colleagues have already done startup checks without going to the glove station. The blade is stationary during alignment checks.",
         "Time lost per glove-station trip: ~2 minutes. Shift already behind: 30 minutes.",
         ["Go to the glove station, get the gloves, do the check correctly regardless of time cost.",
          "Do the check without gloves but formally note in the shift log that gloves were unavailable.",
          "Do the check without gloves. The blade is off and the policy was designed for active blade work not static alignment.",
          "Ask the team lead whether the glove requirement applies to static pre-startup alignment, and act on their answer.",
          "Do the check without gloves but raise the glove station placement issue with the supervisor after the shift."]),
        ("C1", "Clerical",
         "You are processing monthly payroll, a task you have done every month for two years. You are on entry 39 of 60 when a colleague asks an urgent question. The conversation takes six minutes. When you return your best estimate is entry 39 but you minimised the window and are not certain. One transposed account number causes a formal payroll error report. Deadline in 38 minutes.",
         "Submission deadline: 38 minutes. Verification from entry 35: ~14 minutes.",
         ["Go back to entry 35 and recheck from there before continuing, accepting the tight deadline.",
          "Continue from entry 39. You have done this two years and your memory is reliable enough.",
          "Continue from 39, submit on time, immediately flag to your manager for a spot-check of entries 35–40.",
          "Quickly scan entries 37–40 against paper forms for any obvious mismatch, continue if nothing looks wrong.",
          "Email your manager explaining the situation and ask whether to submit on time or delay for full verification."]),
        ("C2", "Clerical",
         "You are reviewing loan applications, on application 44 of 60. This application has a mismatch: the employer name on the form does not match the income verification document. You have seen this exact pattern eight or nine times this year. Every single time it was a recent employer name change and the application was approved. Flagging triggers a 3-day compliance hold. Your team lead said the backlog is a concern today.",
         "Applications remaining: 16. Flagging triggers: 3-day compliance hold. Previous similar cases: all resolved.",
         ["Flag it for compliance review. The policy exists for situations like this regardless of past pattern.",
          "Approve it. Every previous case with this pattern was legitimate and the other data points all match.",
          "Call the applicant directly to clarify the employer name discrepancy before deciding, and document the call.",
          "Approve it but add an internal note documenting the mismatch, your reasoning, and the pattern from previous cases.",
          "Escalate to your team lead: given the consistent pattern and backlog ask whether flagging or approving with documentation is preferred."]),
        ("C3", "Clerical",
         "It is 4:41 PM. You receive an email from the Head of Operations, two levels above you: Please send me the full compensation breakdown file for the retail division immediately. Board meeting starts at 5 PM. The sender's name, title, and email match your directory. However the request is not routed through the HR data-sharing system required by data governance policy. Formal access request takes 25 minutes. Your manager left early and is unreachable.",
         "Time until meeting: 19 minutes. Formal access request: 25 minutes. Your manager: unreachable.",
         ["Reply explaining you cannot release data outside the formal system and offer to initiate the request immediately, copying HR compliance.",
          "Send the file. The request is from a director, time-critical, and clearly from the right person. Withholding would be obstruction.",
          "Call the Head of Operations' office phone directly to verbally confirm the request before sending anything.",
          "Forward to the HR compliance mailbox and ask whether an emergency override exists for director-level board requests.",
          "Send a summary version with aggregate figures but without individual salary data as a compromise."]),
    ]

    with st.form(key="page_5_form"):
        scenario_choices = {}
        scenario_rp      = {}
        scenario_conf    = {}

        for sid, stype, vignette, pressure, options in scenarios:
            color = "🔵" if stype == "Manufacturing" else "🟢"
            st.subheader(f"{color} Scenario {sid} — {stype}")
            st.write(vignette)
            st.warning(pressure)

            choice = st.radio(
                "What do you do?",
                options,
                key=f"sc_{sid}",
                index=None
            )
            rp_val = st.select_slider(
                "How risky is your chosen action?",
                options=[1, 2, 3, 4, 5],
                value=3,
                key=f"rp_{sid}",
                format_func=lambda x: {1: "1 Not risky", 2: "2", 3: "3 Moderate", 4: "4", 5: "5 Extremely risky"}[x]
            )
            sc_val = st.select_slider(
                "How confident are you your choice was correct?",
                options=[1, 2, 3, 4, 5],
                value=3,
                key=f"sc_conf_{sid}",
                format_func=lambda x: {1: "1 Not confident", 2: "2", 3: "3 Moderate", 4: "4", 5: "5 Fully confident"}[x]
            )
            scenario_choices[sid] = choice
            scenario_rp[sid]      = rp_val
            scenario_conf[sid]    = sc_val
            st.divider()

        submitted = st.form_submit_button("Get Results →", type="primary")
        if submitted:
            if None in scenario_choices.values():
                st.error("Please answer all 6 scenarios before continuing.")
            else:
                codes, unsafe_flags = [], []
                for sid, stype, vignette, pressure, opts_list in scenarios:
                    chosen = scenario_choices[sid]
                    idx    = opts_list.index(chosen)
                    code   = rubric[sid][idx]
                    codes.append(code)
                    unsafe_flags.append(1 if code > 0 else 0)

                st.session_state.data['unsafe_total']    = sum(unsafe_flags)
                st.session_state.data['scenario_codes']  = codes
                st.session_state.data['risk_perception'] = np.mean(
                    [scenario_rp[s] for s in scenario_rp])
                st.session_state.data['scenario_rp']     = scenario_rp
                st.session_state.data['scenario_conf']   = scenario_conf
                next_page()
                st.rerun()

# ══════════════════════════════════════════════════════════
# PAGE 6 — RESULTS
# ══════════════════════════════════════════════════════════
elif st.session_state.page == 6:
    d = st.session_state.data

    # All variables defined at the top — prevents NameError
    imp  = d['impulsivity']
    str_ = d['stress']
    oc   = d['overconfidence']
    rp   = d['risk_perception']
    jt   = d['job_type']

    imp_z = z_score(imp,  bench['impulsivity_mean'],     bench['impulsivity_std'])
    str_z = z_score(str_, bench['stress_mean'],          bench['stress_std'])
    oc_z  = z_score(oc,   bench['overconfidence_mean'],  bench['overconfidence_std'])
    rp_z  = z_score(rp,   bench['risk_perception_mean'], bench['risk_perception_std'])

    unsafe_count    = sum(1 for c in d['scenario_codes'] if c > 0)
    violation_count = sum(1 for c in d['scenario_codes'] if c == 3)
    error_count     = sum(1 for c in d['scenario_codes'] if c == 2)
    partial_count   = sum(1 for c in d['scenario_codes'] if c == 1)
    safe_count      = sum(1 for c in d['scenario_codes'] if c == 0)

    # ── Predict ───────────────────────────────────────────
    input_arr   = scaler.transform([[imp, str_, oc, rp, jt]])
    prediction  = model.predict(input_arr)[0]
    prob_unsafe = model.predict_proba(input_arr)[0][1]
    prob_safe   = 1 - prob_unsafe

    st.header("Your Decision Risk Profile")

    # ── Main result ───────────────────────────────────────
    st.subheader("Overall Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        if prediction == 1:
            st.error("Higher Risk Pattern")
        else:
            st.success("Lower Risk Pattern")
    with col2:
        st.metric("Unsafe probability", f"{prob_unsafe * 100:.1f}%")
    with col3:
        st.metric("Safe probability", f"{prob_safe * 100:.1f}%")

    st.divider()

    # ── Your Scores ───────────────────────────────────────
    st.subheader("Your Scores")
    score_col1, score_col2, score_col3, score_col4, score_col5 = st.columns(5)
    with score_col1:
        st.metric("Impulsivity", f"{imp:.0f}",
                  delta=f"{imp - bench['impulsivity_mean']:.1f} vs avg",
                  delta_color="inverse")
    with score_col2:
        st.metric("Stress", f"{str_:.0f}",
                  delta=f"{str_ - bench['stress_mean']:.1f} vs avg",
                  delta_color="inverse")
    with score_col3:
        st.metric("Overconfidence", f"{oc:.1f}%",
                  delta=f"{oc - bench['overconfidence_mean']:.1f} vs avg",
                  delta_color="inverse")
    with score_col4:
        st.metric("Risk Perception", f"{rp:.2f}",
                  delta=f"{rp - bench['risk_perception_mean']:.2f} vs avg",
                  delta_color="normal")
    with score_col5:
        st.metric("Unsafe decisions", f"{unsafe_count}/6")

    st.divider()

    # ── Scenario breakdown ────────────────────────────────
    st.subheader("Scenario Response Breakdown")
    scenario_ids = ['M1', 'M2', 'M3', 'C1', 'C2', 'C3']
    code_labels  = {0: "Safe", 1: "Partial", 2: "Unsafe Error", 3: "Violation"}
    code_colors  = {0: "green", 1: "orange", 2: "red", 3: "darkred"}

    cols = st.columns(6)
    for i, (sid, code) in enumerate(zip(scenario_ids, d['scenario_codes'])):
        with cols[i]:
            label = code_labels[code]
            color = code_colors[code]
            st.markdown(f"**{sid}**")
            st.markdown(f":{color}[{label}]")

    st.write(f"**{unsafe_count} of 6 scenarios** resulted in unsafe or partially unsafe decisions.")

    st.divider()

    # ── Profile vs benchmark ──────────────────────────────
    st.subheader("Your Profile vs Study Sample")

    profile = {
        "Impulsivity":     (imp,  bench['impulsivity_mean'],     bench['impulsivity_std']),
        "Stress":          (str_, bench['stress_mean'],          bench['stress_std']),
        "Overconfidence":  (oc,   bench['overconfidence_mean'],  bench['overconfidence_std']),
        "Risk perception": (rp,   bench['risk_perception_mean'], bench['risk_perception_std']),
    }

    for factor, (val, mean, std) in profile.items():
        z = z_score(val, mean, std)
        if z > 0.5:
            level, color = "Above average", "red"
        elif z < -0.5:
            level, color = "Below average", "green"
        else:
            level, color = "Average", "blue"

        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.write(f"**{factor}**")
        with col2:
            st.write(f"{val:.1f}")
        with col3:
            st.markdown(f":{color}[{level} (study mean: {mean:.1f})]")

    st.divider()

    # ── What your profile means ───────────────────────────
    st.subheader("What Your Profile Means")

    if imp_z > 0.5:
        with st.expander("⚡ High Impulsivity — What this means for you"):
            st.write(f"""
            **Your score: {imp:.0f} | Study average: {bench['impulsivity_mean']:.0f}**

            Your BIS-11 score places you above average on impulsivity, reflecting a
            tendency to act before fully evaluating consequences. Reason's framework
            identifies this as a precursor to skill-based slips and violations — not
            because you are careless, but because your cognitive system moves quickly
            from intention to action without a sufficient pause for risk evaluation.

            **What to do:**
            - Before acting in any uncertain situation, use a deliberate 3-second pause.
              Ask: have I done this specific check today?
            - Use physical checklists for routine tasks.
            - In high-pressure moments, treat the urge to act quickly as a signal to
              slow down, not speed up.
            """)
    elif imp_z < -0.5:
        with st.expander("✅ Low Impulsivity — What this means for you"):
            st.write(f"""
            **Your score: {imp:.0f} | Study average: {bench['impulsivity_mean']:.0f}**

            Your impulsivity score is below average, suggesting you tend to think before
            acting. This is a protective factor in workplace safety. Be aware that low
            stress can also produce complacency in routine tasks.
            """)

    if str_z > 0.5:
        with st.expander("😰 Elevated Stress — What this means for you"):
            st.write(f"""
            **Your score: {str_:.0f} | Study average: {bench['stress_mean']:.0f}**

            Your PSS-10 score indicates elevated perceived stress. Stress consumes the
            limited cognitive capacity available for deliberate thinking. When you are
            stressed, attentional bandwidth narrows to the most immediate goal and
            peripheral risk signals fall outside that window.

            **What to do:**
            - Protect your attention during high-consequence tasks — no interruptions
              during payroll entry, safety checks, or compliance reviews.
            - When stress is high, verification effort goes up, not down.
            - Communicate overload to your supervisor before it affects decision quality.
            """)
    elif str_z < -0.5:
        with st.expander("✅ Low Stress — What this means for you"):
            st.write(f"""
            **Your score: {str_:.0f} | Study average: {bench['stress_mean']:.0f}**

            Your perceived stress is below average — a meaningful protective factor.
            Be aware that low stress can sometimes reduce vigilance in routine tasks
            through complacency.
            """)

    if oc_z > 0.5:
        with st.expander("🎯 Overconfidence Detected — What this means for you"):
            st.write(f"""
            **Your gap: {oc:.1f}% | Study average: {bench['overconfidence_mean']:.1f}%**

            Your confidence exceeded your actual accuracy by {oc:.1f} percentage points.
            This is a systematic miscalibration that reduces the perceived need to verify.
            If you feel certain your interpretation is correct, you will not seek
            additional information or pause to check — even when you should.

            **What to do:**
            - Treat your first interpretation as a hypothesis, not a conclusion.
            - Ask: what would have to be true for this to be wrong?
            - When you feel most certain, that is exactly when verification matters most.
            """)
    elif oc_z < -0.5:
        with st.expander("✅ Well-Calibrated Confidence — What this means for you"):
            st.write(f"""
            **Your gap: {oc:.1f}% | Study average: {bench['overconfidence_mean']:.1f}%**

            Your confidence closely matched your actual accuracy — a protective factor
            in decision-making. If your gap is negative you may be slightly
            underconfident; trust your verified judgment when evidence supports it.
            """)

    if rp_z < -0.5:
        with st.expander("⚠️ Low Risk Perception — What this means for you"):
            st.write(f"""
            **Your score: {rp:.2f} | Study average: {bench['risk_perception_mean']:.2f}**

            Across the six scenarios you consistently rated your chosen actions as low
            risk. Low risk perception often reflects familiarity bias, a controllability
            illusion, or abstract consequence blindness.

            **What to do:**
            - When a situation feels familiar and low-risk, ask: what is different
              about this specific instance?
            - Pay deliberate attention to abstract and delayed consequences.
            - If safe behaviour requires effort and the unsafe option feels natural,
              that asymmetry is itself a warning sign.
            """)

    st.divider()

    # ── Scenario decision pattern ─────────────────────────
    with st.expander("📊 Your Scenario Decision Pattern"):
        st.write(f"""
        **Safe decisions: {safe_count}/6**
        **Partial / ambiguous: {partial_count}/6**
        **Unsafe errors (unintentional): {error_count}/6**
        **Violations (deliberate bypass): {violation_count}/6**
        """)

        if violation_count > 1:
            st.write(f"""
            **{violation_count} of your responses were coded as violations** — deliberate
            bypasses of a known rule or procedure. The intervention is not more knowledge
            of the rule. You already know it. The intervention is changing the perceived
            cost of compliance — making the safe option easier, faster, or more socially
            supported.
            """)

        if error_count > 1:
            st.write(f"""
            **{error_count} of your responses reflected unintentional unsafe errors.**
            The primary drivers are attention failure, routine automaticity, and cognitive
            narrowing under stress. Forcing functions — system checks, mandatory
            verification steps, physical checklists — are more effective interventions
            than awareness training.
            """)

        if safe_count == 6:
            st.write("""
            All six of your scenario responses were coded as safe or fully compliant.
            The key is maintaining this standard under genuine time pressure and authority
            pressure — which is harder in real workplaces than in scenario tasks.
            """)

    st.divider()

    # ── Job context ───────────────────────────────────────
    jt_label = "Manufacturing" if jt == 1 else "Clerical"
    st.subheader(f"Your Context: {jt_label}")

    if jt == 1:
        st.write("""
        In manufacturing environments, unsafe decisions have immediate physical
        consequences. Production pressure from supervisors is the single strongest
        predictor of unsafe behaviour in manufacturing contexts, overriding individual
        safety knowledge. Your primary safeguard is procedural discipline: follow the
        check, wear the PPE, escalate the warning — regardless of what the production
        clock says.
        """)
    else:
        st.write("""
        In clerical and administrative environments, unsafe decisions produce delayed,
        abstract consequences — data breaches, compliance failures, financial errors.
        Abstract and delayed risks are chronically underestimated relative to their
        actual severity. Your primary safeguard is verification discipline — flag the
        ambiguity, follow the data governance protocol, escalate the unusual request
        regardless of who sent it.
        """)

    st.divider()

    if st.button("Start over", type="primary"):
        st.session_state.page = 0
        st.session_state.data = {}
        st.rerun()
