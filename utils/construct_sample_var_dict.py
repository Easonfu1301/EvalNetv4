import numpy as np
import numba
from numba import jit


def construct_sample_var_dict(samples, hit_dict):
    # print(samples)
    # print(hit_dict)

    xyz_l0 = np.array([hit_dict[l0][:3] for l0 in samples[:, 0]])
    xyz_l1 = np.array([hit_dict[l1][:3] for l1 in samples[:, 1]])
    xyz_l2 = np.array([hit_dict[l2][:3] for l2 in samples[:, 2]])
    xyz_l3 = np.array([hit_dict[l3][:3] for l3 in samples[:, 3]])

    n1 = np.array([hit_dict[l1][3:4] for l1 in samples[:, 0]])
    n2 = np.array([hit_dict[l2][3:4] for l2 in samples[:, 1]])
    n3 = np.array([hit_dict[l3][3:4] for l3 in samples[:, 2]])

    # samples_kind = samples[:, 4]

    cos_theta01 = np.sum(xyz_l0 * xyz_l1, axis=1) / (np.linalg.norm(xyz_l0, axis=1) * np.linalg.norm(xyz_l1, axis=1))
    cos_theta02 = np.sum(xyz_l0 * xyz_l2, axis=1) / (np.linalg.norm(xyz_l0, axis=1) * np.linalg.norm(xyz_l2, axis=1))
    cos_theta03 = np.sum(xyz_l0 * xyz_l3, axis=1) / (np.linalg.norm(xyz_l0, axis=1) * np.linalg.norm(xyz_l3, axis=1))

    cos_theta12 = np.sum(xyz_l1 * xyz_l2, axis=1) / (np.linalg.norm(xyz_l1, axis=1) * np.linalg.norm(xyz_l2, axis=1))
    cos_theta13 = np.sum(xyz_l1 * xyz_l3, axis=1) / (np.linalg.norm(xyz_l1, axis=1) * np.linalg.norm(xyz_l3, axis=1))

    cos_theta23 = np.sum(xyz_l2 * xyz_l3, axis=1) / (np.linalg.norm(xyz_l2, axis=1) * np.linalg.norm(xyz_l3, axis=1))

    cos_theta01[cos_theta01 > 1] = 1
    cos_theta02[cos_theta02 > 1] = 1
    cos_theta03[cos_theta03 > 1] = 1
    cos_theta12[cos_theta12 > 1] = 1
    cos_theta13[cos_theta13 > 1] = 1
    cos_theta23[cos_theta23 > 1] = 1

    theta01 = np.arccos(cos_theta01)
    theta02 = np.arccos(cos_theta02)
    theta03 = np.arccos(cos_theta03)
    theta12 = np.arccos(cos_theta12)
    theta13 = np.arccos(cos_theta13)
    theta23 = np.arccos(cos_theta23)



    data = np.concatenate([theta01.reshape(-1, 1), theta02.reshape(-1, 1), theta03.reshape(-1, 1),
                           theta12.reshape(-1, 1), theta13.reshape(-1, 1), theta23.reshape(-1, 1), n1, n2, n3], axis=1)
    # print(data.shape)

    return data