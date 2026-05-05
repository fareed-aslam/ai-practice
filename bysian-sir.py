from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('CameraFault', 'TrafficAlert'),
    ('RadarFault', 'TrafficAlert'),
    ('LoopFault', 'TrafficAlert'),
    ('TrafficAlert', 'ControlRoomNotif'),
    ('TrafficAlert', 'MobileAlert')
])

cpd_camera = TabularCPD('CameraFault', 2, [[0.94], [0.06]])
cpd_radar = TabularCPD('RadarFault', 2, [[0.91], [0.09]])
cpd_loop = TabularCPD('LoopFault', 2, [[0.93], [0.07]])

cpd_alert = TabularCPD(
    'TrafficAlert', 2,
    [
        [0.99, 0.90, 0.90, 0.20, 0.90, 0.20, 0.20, 0.01],
        [0.01, 0.10, 0.10, 0.80, 0.10, 0.80, 0.80, 0.99]
    ],
    ['CameraFault', 'RadarFault', 'LoopFault'],
    [2, 2, 2]
)

cpd_control = TabularCPD(
    'ControlRoomNotif', 2,
    [
        [0.03, 0.97],
        [0.97, 0.03]
    ],
    ['TrafficAlert'],
    [2]
)

cpd_mobile = TabularCPD(
    'MobileAlert', 2,
    [
        [0.96, 0.12],
        [0.04, 0.88]
    ],
    ['TrafficAlert'],
    [2]
)

model.add_cpds(cpd_camera, cpd_radar, cpd_loop, cpd_alert, cpd_control, cpd_mobile)
model.check_model()

infer = VariableElimination(model)

print(infer.query(['TrafficAlert'], evidence={'ControlRoomNotif': 1, 'MobileAlert': 1}))
print(infer.query(['RadarFault'], evidence={'TrafficAlert': 1}))
