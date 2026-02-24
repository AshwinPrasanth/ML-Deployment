# ======================================================
# ADVANCED MULTI-SCENARIO EVALUATION
# ======================================================

SCENARIOS = {

    "Cardiac + Diabetes": {
        "profile": {
            "age": 55,
            "chronic_conditions": ["heart_disease", "diabetes"]
        },
        "query": """
        Cardiac procedures, insulin support,
        frequent inpatient admission,
        consultant coverage, high-tech hospital.
        """
    },

    "Cancer + Neurological": {
        "profile": {
            "age": 60,
            "chronic_conditions": ["cancer", "neurological_disorder"]
        },
        "query": """
        Oncology treatment, chemotherapy,
        MRI scans, neurological admission,
        high-tech hospital access.
        """
    },

    "Psychiatric + Diabetes": {
        "profile": {
            "age": 40,
            "chronic_conditions": ["psychiatric_disorder", "diabetes"]
        },
        "query": """
        Psychiatric admission days,
        mental health coverage,
        diabetes management.
        """
    },

    "Pregnancy Case": {
        "profile": {
            "age": 30,
            "chronic_conditions": ["pregnancy"]
        },
        "query": """
        Maternity cover,
        obstetric services,
        hospital delivery,
        minimal excess.
        """
    },

    "High-Risk Elderly Multi-Morbidity": {
        "profile": {
            "age": 72,
            "chronic_conditions": ["heart_disease", "cancer", "neurological_disorder"]
        },
        "query": """
        Cardiac procedures, chemotherapy,
        MRI scans, frequent hospital admissions,
        high-tech hospital access.
        """
    },

    "Long-Term Psychiatric Intensive": {
        "profile": {
            "age": 42,
            "chronic_conditions": ["psychiatric_disorder"]
        },
        "query": """
        Extended psychiatric inpatient treatment,
        high number of covered days,
        strong mental health support.
        """
    }
}

print("\n====================================")
print("ADVANCED MULTI-SCENARIO EVALUATION (DENSE ONLY)")
print("====================================")

plan_win_counter = defaultdict(int)
plan_rank_sum = defaultdict(int)
scenario_margins = []

for scenario_name, scenario_data in SCENARIOS.items():

    print("\n==============================")
    print(f"Scenario: {scenario_name}")
    print("==============================")

    patient_profile = scenario_data["profile"]
    query = scenario_data["query"]

    dense_scores = compute_dense_scores(query)

    results = []

    for plan_id in plan_ids:

        risk_score, breakdown = compute_risk_score(plan_id, patient_profile)

        final_score = (BETA * risk_score) + dense_scores.get(plan_id, 0)

        results.append((plan_id, final_score, risk_score,
                        dense_scores.get(plan_id, 0), breakdown))

    results.sort(key=lambda x: x[1], reverse=True)

    # Track winner
    winner = results[0][0]
    plan_win_counter[winner] += 1

    # Track average ranks
    for rank, (doc_id, *_ ) in enumerate(results, 1):
        plan_rank_sum[doc_id] += rank

    # Print top results
    for rank, (doc_id, final_score, risk_score,
               dense_score, breakdown) in enumerate(results[:FINAL_TOP_K], 1):

        print(f"\n{rank}. {doc_id}")
        print(f"   Final Score: {round(final_score,2)}")
        print(f"   Risk Score: {round(risk_score,2)}")
        print(f"   Dense Score: {round(dense_score,3)}")

        print("   Breakdown:")
        for k, v in breakdown.items():
            print(f"     {k}: {v}")

    margin = results[0][1] - results[1][1]
    scenario_margins.append(margin)

    print("\nConfidence Margin:", round(margin, 3))

    if margin < 1:
        print("Interpretation: Close competition")
    elif margin < 3:
        print("Interpretation: Moderate separation")
    else:
        print("Interpretation: Strong winner")

# ======================================================
# GLOBAL ANALYTICS
# ======================================================

print("\n====================================")
print("GLOBAL ANALYTICS")
print("====================================")

print("\nPlan Win Frequency:")
for plan, count in sorted(plan_win_counter.items(), key=lambda x: x[1], reverse=True):
    print(f"{plan}: {count} wins")

print("\nAverage Rank Per Plan:")
num_scenarios = len(SCENARIOS)

avg_ranks = {
    plan: plan_rank_sum[plan] / num_scenarios
    for plan in plan_ids
}

for plan, avg_rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
    print(f"{plan}: Avg Rank = {round(avg_rank,2)}")

print("\nAverage Confidence Margin Across Scenarios:",
      round(np.mean(scenario_margins), 3))

print("Margin Std Dev:",
      round(np.std(scenario_margins), 3))
