# ============================
# SCENARIO EVALUATION
# ============================

SCENARIOS = [

    {
        "name": "High-Risk Cardiac + Diabetes",
        "profile": {
            "age": 55,
            "chronic_conditions": ["heart_disease", "diabetes"],
            "medication_frequency": "daily",
            "specialist_visits_per_year": 6,
            "hospital_admissions_last_2_years": 2
        }
    },

    {
        "name": "Cancer Patient",
        "profile": {
            "age": 48,
            "chronic_conditions": ["cancer"],
            "medication_frequency": "daily",
            "specialist_visits_per_year": 8,
            "hospital_admissions_last_2_years": 3
        }
    },

    {
        "name": "Maternity Case",
        "profile": {
            "age": 32,
            "chronic_conditions": ["pregnancy"],
            "medication_frequency": "monthly",
            "specialist_visits_per_year": 4,
            "hospital_admissions_last_2_years": 0
        }
    },

    {
        "name": "Psychiatric Condition",
        "profile": {
            "age": 40,
            "chronic_conditions": ["psychiatric_disorder"],
            "medication_frequency": "daily",
            "specialist_visits_per_year": 5,
            "hospital_admissions_last_2_years": 1
        }
    },

    {
        "name": "Young Low-Risk Adult",
        "profile": {
            "age": 25,
            "chronic_conditions": [],
            "medication_frequency": "none",
            "specialist_visits_per_year": 1,
            "hospital_admissions_last_2_years": 0
        }
    },

    {
        "name": "Neurological Disorder",
        "profile": {
            "age": 60,
            "chronic_conditions": ["neurological_disorder"],
            "medication_frequency": "daily",
            "specialist_visits_per_year": 7,
            "hospital_admissions_last_2_years": 2
        }
    }
]

print("\n==============================")
print("SCENARIO EVALUATION")
print("==============================")

for scenario in SCENARIOS:

    name = scenario["name"]
    profile = scenario["profile"]

    ranked = sorted(
        plans,
        key=lambda p: score_plan(p, profile),
        reverse=True
    )

    print(f"\n--- Scenario: {name} ---")

    for i, plan in enumerate(ranked[:3], 1):
        print(f"{i}. {plan['doc_id']}")
        print(f"   Score: {score_plan(plan, profile)}")
        print(f"   Excess: €{plan['semi_private_excess']}")
        print(f"   High-Tech %: {plan['high_tech_percent']}")
        print(f"   Psychiatric Days: {plan['psychiatric_days']}")
        print("-" * 50)
