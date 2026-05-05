from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('Intelligence', 'Grade'),
    ('StudyHours', 'Grade'),
    ('Difficulty', 'Grade'),
    ('Grade', 'Pass')
])

cpd_intelligence = TabularCPD(
    variable='Intelligence',
    variable_card=2,
    values=[[0.3], [0.7]]
)

cpd_study = TabularCPD(
    variable='StudyHours',
    variable_card=2,
    values=[[0.4], [0.6]]
)

cpd_difficulty = TabularCPD(
    variable='Difficulty',
    variable_card=2,
    values=[[0.4], [0.6]]
)

cpd_grade = TabularCPD(
    variable='Grade',
    variable_card=3,
    values=[
        [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.08, 0.2, 0.25, 0.3, 0.4, 0.4, 0.5, 0.4],
        [0.02, 0.1, 0.15, 0.2, 0.2, 0.3, 0.3, 0.5]
    ],
    evidence=['Intelligence', 'StudyHours', 'Difficulty'],
    evidence_card=[2, 2, 2]
)

cpd_pass = TabularCPD(
    variable='Pass',
    variable_card=2,
    values=[
        [0.05, 0.20, 0.50],
        [0.95, 0.80, 0.50]
    ],
    evidence=['Grade'],
    evidence_card=[3]
)

model.add_cpds(cpd_intelligence, cpd_study, cpd_difficulty, cpd_grade, cpd_pass)

model.check_model()

infer = VariableElimination(model)

result1 = infer.query(
    variables=['Pass'],
    evidence={'StudyHours': 1, 'Difficulty': 0}
)

print(result1)

result2 = infer.query(
    variables=['Intelligence'],
    evidence={'Pass': 1}
)

print(result2)
