"""
Data provided by
Chang, J.Y., Moon, G., Lee, K.M.: Absposelifter: Absolute 3d human pose lifting network from a single noisy 2d human pose. arXiv preprint arXiv:1910.12029 (2020)
"""
error_distribution = [
dict(
Joint = 'Pelvis',
mean = (-0.06, -2.37),
std = (1.33, 2.13),
weight = (1.00)
),
dict(
Joint = 'R_Hip',
mean = (-0.83, -2.07),
std = (3.41, 2.69),
weight = (1.00)
)
,
dict(
Joint = 'R_Knee',
mean = (-0.04, -1.01),
std = (1.74, 2.20),
weight = (0.95)
)
,
dict(
Joint = 'R_Ankle',
mean = (0.52, -3.40),
std = (1.39, 2.14),
weight = (0.93)
)
,
dict(
Joint = 'L_Hip',
mean = (0.78, -2.79),
std = (3.26, 2.28),
weight = (1.00)
)
,
dict(
Joint = 'L_Knee',
mean = (0.42, -0.15),
std = (1.53, 1.99),
weight = (0.94)
)
,
dict(
Joint = 'L_Ankle',
mean = (-0.15, -3.78),
std = (1.39, 2.39),
weight = (0.93)
)
,
dict(
Joint = 'Torso',
mean = (-0.05, 0.10),
std = (1.36, 1.74),
weight = (0.99)
)
,
dict(
Joint = 'Neck',
mean = (0.14, -2.56),
std = (1.18, 1.15),
weight = (0.99)
)
,
dict(
Joint = 'Head',
mean = (0.09, 0.49),
std = (1.35, 0.87),
weight = (0.99)
)
,
dict(
Joint = 'Nose',
mean = (0.13, -0.26),
std = (0.78, 0.59),
weight = (0.98)
)
,
dict(
Joint = 'L_Shoulder',
mean = (-0.19, 0.31),
std = (2.51, 1.48),
weight = (0.99)
)
,
dict(
Joint = 'L_Elbow',
mean = (0.11, -0.60),
std = (1.79, 1.76),
weight = (0.95)
)
,
dict(
Joint = 'L_Wrist',
mean = (-0.02, 0.88),
std = (2.02, 2.10),
weight = (0.91)
)
,
dict(
Joint = 'R_Shoulder',
mean = (0.52, -0.12),
std = (2.23, 1.73),
weight = (0.99)
)
,
dict(
Joint = 'R_Elbow',
mean = (0.06, -0.44),
std = (1.93, 1.63),
weight = (0.95)
)
,
dict(
Joint = 'R_Wrist',
mean = (0.05, 0.16),
std = (2.02, 2.24),
weight = (0.90)
)
]
