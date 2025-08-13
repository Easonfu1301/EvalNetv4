import numpy as np



# def construct_sample_var(samples, hit_df):
#     # print(samples)
#     xyz_l0 = hit_df.loc[samples[:, 0], ["x", "y", "z"]].values
#     xyz_l1 = hit_df.loc[samples[:, 1], ["x", "y", "z"]].values
#     xyz_l2 = hit_df.loc[samples[:, 2], ["x", "y", "z"]].values
#     xyz_l3 = hit_df.loc[samples[:, 3], ["x", "y", "z"]].values
#
#     # samples_kind = samples[:, 4]
#
#     cos_theta01 = np.sum(xyz_l0 * xyz_l1, axis=1) / (np.linalg.norm(xyz_l0, axis=1) * np.linalg.norm(xyz_l1, axis=1))
#     cos_theta02 = np.sum(xyz_l0 * xyz_l2, axis=1) / (np.linalg.norm(xyz_l0, axis=1) * np.linalg.norm(xyz_l2, axis=1))
#     cos_theta03 = np.sum(xyz_l0 * xyz_l3, axis=1) / (np.linalg.norm(xyz_l0, axis=1) * np.linalg.norm(xyz_l3, axis=1))
#
#     cos_theta12 = np.sum(xyz_l1 * xyz_l2, axis=1) / (np.linalg.norm(xyz_l1, axis=1) * np.linalg.norm(xyz_l2, axis=1))
#     cos_theta13 = np.sum(xyz_l1 * xyz_l3, axis=1) / (np.linalg.norm(xyz_l1, axis=1) * np.linalg.norm(xyz_l3, axis=1))
#
#     cos_theta23 = np.sum(xyz_l2 * xyz_l3, axis=1) / (np.linalg.norm(xyz_l2, axis=1) * np.linalg.norm(xyz_l3, axis=1))
#
#     cos_theta01[cos_theta01 > 1] = 1
#     cos_theta02[cos_theta02 > 1] = 1
#     cos_theta03[cos_theta03 > 1] = 1
#     cos_theta12[cos_theta12 > 1] = 1
#     cos_theta13[cos_theta13 > 1] = 1
#     cos_theta23[cos_theta23 > 1] = 1
#
#     theta01 = np.arccos(cos_theta01)
#     theta02 = np.arccos(cos_theta02)
#     theta03 = np.arccos(cos_theta03)
#     theta12 = np.arccos(cos_theta12)
#     theta13 = np.arccos(cos_theta13)
#     theta23 = np.arccos(cos_theta23)
#
#     n1 = hit_df.loc[samples[:, 0], ["nearest_dist"]].values
#     n2 = hit_df.loc[samples[:, 1], ["nearest_dist"]].values
#     n3 = hit_df.loc[samples[:, 2], ["nearest_dist"]].values
#
#     data = np.concatenate([theta01.reshape(-1, 1), theta02.reshape(-1, 1), theta03.reshape(-1, 1),
#                            theta12.reshape(-1, 1), theta13.reshape(-1, 1), theta23.reshape(-1, 1), n1, n2, n3], axis=1)
#     # print(data.shape)
#
#     return data



def construct_sample_var(samples, hit_df):
    # print(samples)

    xyz_n_l0 = hit_df.loc[samples[:, 0], ["x", "y", "z", "nearest_dist"]].values
    xyz_n_l1 = hit_df.loc[samples[:, 1], ["x", "y", "z", "nearest_dist"]].values
    xyz_n_l2 = hit_df.loc[samples[:, 2], ["x", "y", "z", "nearest_dist"]].values
    xyz_n_l3 = hit_df.loc[samples[:, 3], ["x", "y", "z"]].values



    xyz_l0 = xyz_n_l0[:, :3]
    xyz_l1 = xyz_n_l1[:, :3]
    xyz_l2 = xyz_n_l2[:, :3]
    xyz_l3 = xyz_n_l3[:, :3]

    # samples_kind = samples[:, 4]

    norm_l0 = np.linalg.norm(xyz_l0, axis=1)
    norm_l1 = np.linalg.norm(xyz_l1, axis=1)
    norm_l2 = np.linalg.norm(xyz_l2, axis=1)
    norm_l3 = np.linalg.norm(xyz_l3, axis=1)



    cos_theta01 = np.sum(xyz_l0 * xyz_l1, axis=1) / (norm_l0 * norm_l1)
    cos_theta02 = np.sum(xyz_l0 * xyz_l2, axis=1) / (norm_l0 * norm_l2)
    cos_theta03 = np.sum(xyz_l0 * xyz_l3, axis=1) / (norm_l0 * norm_l3)

    cos_theta12 = np.sum(xyz_l1 * xyz_l2, axis=1) / (norm_l1 * norm_l2)
    cos_theta13 = np.sum(xyz_l1 * xyz_l3, axis=1) / (norm_l1 * norm_l3)

    cos_theta23 = np.sum(xyz_l2 * xyz_l3, axis=1) / (norm_l2 * norm_l3)

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

    n1 = xyz_n_l0[:, 3:4]
    n2 = xyz_n_l1[:, 3:4]
    n3 = xyz_n_l2[:, 3:4]


    data = np.concatenate([theta01.reshape(-1, 1), theta02.reshape(-1, 1), theta03.reshape(-1, 1),
                           theta12.reshape(-1, 1), theta13.reshape(-1, 1), theta23.reshape(-1, 1), n1, n2, n3], axis=1)
    # print(data.shape)

    return data