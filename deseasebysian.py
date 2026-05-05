from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('Disease', 'Fever'),
    ('Disease', 'Cough'),
    ('Disease', 'Fatigue'),
    ('Disease', 'Chills')
])

cpd_disease = TabularCPD('Disease', 2, [[0.3], [0.7]])

cpd_fever = TabularCPD('Fever', 2, [[0.1, 0.5], [0.9, 0.5]], ['Disease'], [2])
cpd_cough = TabularCPD('Cough', 2, [[0.2, 0.4], [0.8, 0.6]], ['Disease'], [2])
cpd_fatigue = TabularCPD('Fatigue', 2, [[0.3, 0.7], [0.7, 0.3]], ['Disease'], [2])
cpd_chills = TabularCPD('Chills', 2, [[0.4, 0.6], [0.6, 0.4]], ['Disease'], [2])

model.add_cpds(cpd_disease, cpd_fever, cpd_cough, cpd_fatigue, cpd_chills)

model.check_model()

infer = VariableElimination(model)

print(infer.query(['Disease'], evidence={'Fever': 1, 'Cough': 1}))
print(infer.query(['Disease'], evidence={'Fever': 1, 'Cough': 1, 'Chills': 1}))
print(infer.query(['Fatigue'], evidence={'Disease': 0}))
