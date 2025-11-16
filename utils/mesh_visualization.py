import numpy as np
import polyscope as ps


# 3d visualization of mapped function
def threeD_plot_mapped_func(vertices1, faces1, vertices2, faces2,
                            f1, mapped_f1_gt, mapped_f1_refined, mapped_f1_init, space=100):
    ps.remove_all_structures()
    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    # from left to right: M1, Initial (M2), FRIDU refined (M2), GT (M2)
    vertices_m1 = vertices1[0].cpu().numpy() + [-space, 0, 0]
    M1 = ps.register_surface_mesh("M1", vertices_m1, faces1[0].cpu().numpy(), enabled=True)
    M1.add_scalar_quantity("f1", f1.squeeze(1).cpu().numpy(), enabled=True)

    vertices_init = vertices2[0].cpu().numpy() + [0, 0, 0]
    M2_init = ps.register_surface_mesh("M2 Init", vertices_init, faces2[0].cpu().numpy(), enabled=True)
    M2_init.add_scalar_quantity("Init_mapped_f1", mapped_f1_init.squeeze(1).cpu().numpy(), enabled=True)

    vertices_refined = vertices2[0].cpu().numpy() + [space * 1, 0, 0]
    M2_refined = ps.register_surface_mesh("M2 Refined", vertices_refined, faces2[0].cpu().numpy(), enabled=True)
    M2_refined.add_scalar_quantity("Refined_mapped_f1", mapped_f1_refined.squeeze(1).cpu().numpy(), enabled=True)

    vertices_gt = vertices2[0].cpu().numpy() + [space*2, 0, 0]
    M2_gt = ps.register_surface_mesh("M2 GT", vertices_gt, faces2[0].cpu().numpy(), enabled=True)
    M2_gt.add_scalar_quantity("GT_mapped_f1", mapped_f1_gt.squeeze(1).cpu().numpy(), enabled=True)

    ps.show()


# def threeD_plot_mapped_scatter(vertices1, faces1, vertices2, faces2,
#                                f1, mapped_f1_gt, mapped_f1_predicted, mapped_f1_computed,
#                                valid_indices_gt, valid_indices_predicted, valid_indices_computed):
#     """
#     Visualizes function values on meshes using scatter plots when values are only available for a subset of vertices.
#
#     Parameters
#     ----------
#     vertices1 : torch.Tensor of shape (1, n1, 3)
#         The vertices of the source mesh.
#     faces1 : torch.Tensor of shape (1, f1, 3)
#         The faces of the source mesh.
#     vertices2 : torch.Tensor of shape (1, n2, 3)
#         The vertices of the target mesh.
#     faces2 : torch.Tensor of shape (1, f2, 3)
#         The faces of the target mesh.
#     f1 : torch.Tensor of shape (n1, 1)
#         Function values on the source mesh.
#     mapped_f1_gt : torch.Tensor of shape (valid_points, 1)
#         Mapped function values from M1 to M2 (ground truth).
#     mapped_f1_predicted : torch.Tensor of shape (valid_points, 1)
#         Mapped function values from M1 to M2 (predicted).
#     mapped_f1_computed : torch.Tensor of shape (valid_points, 1)
#         Mapped function values from M1 to M2 (computed).
#     valid_indices_gt : torch.Tensor of shape (valid_points,)
#         Indices of valid function values for the ground truth mapping.
#     valid_indices_predicted : torch.Tensor of shape (valid_points,)
#         Indices of valid function values for the predicted mapping.
#     valid_indices_computed : torch.Tensor of shape (valid_points,)
#         Indices of valid function values for the computed mapping.
#     """
#
#     # Initialize Polyscope
#     ps.remove_all_structures()
#     ps.init()
#     ps.set_ground_plane_mode("shadow_only")
#
#     space = 100  # Spacing between meshes
#
#     # Convert tensors to numpy
#     vertices_m1 = vertices1[0].cpu().numpy() + [-space, 0, 0]
#     vertices_gt = vertices2[0].cpu().numpy() + [0, 0, 0]
#     vertices_pred = vertices2[0].cpu().numpy() + [space, 0, 0]
#     vertices_computed = vertices2[0].cpu().numpy() + [space * 2, 0, 0]
#
#     faces1_np = faces1[0].cpu().numpy()
#     faces2_np = faces2[0].cpu().numpy()
#
#     # Register meshes
#     M1 = ps.register_surface_mesh("M1", vertices_m1, faces1_np, enabled=True)
#     M2_gt = ps.register_surface_mesh("M2 GT", vertices_gt, faces2_np, enabled=True)
#     M2_pred = ps.register_surface_mesh("M2 Predicted", vertices_pred, faces2_np, enabled=True)
#     M2_computed = ps.register_surface_mesh("M2 Computed", vertices_computed, faces2_np, enabled=True)
#
#     # Add function visualization on M1
#     M1.add_scalar_quantity("f1", f1.squeeze(1).cpu().numpy(), enabled=True)
#
#     # Scatter plot for mapped function values
#     ps.register_point_cloud("GT_mapped_scatter", vertices_gt[valid_indices_gt.cpu().numpy()], enabled=True)\
#       .add_scalar_quantity("GT_mapped_f1", mapped_f1_gt.squeeze(1).cpu().numpy(), enabled=True)
#
#     ps.register_point_cloud("Predicted_mapped_scatter", vertices_pred[valid_indices_predicted.cpu().numpy()], enabled=True)\
#       .add_scalar_quantity("Predicted_mapped_f1", mapped_f1_predicted.squeeze(1).cpu().numpy(), enabled=True)
#
#     ps.register_point_cloud("Computed_mapped_scatter", vertices_computed[valid_indices_computed.cpu().numpy()], enabled=True)\
#       .add_scalar_quantity("Computed_mapped_f1", mapped_f1_computed.squeeze(1).cpu().numpy(), enabled=True)
#
#     ps.show()


# def scatter_landmarks_err(x2, em_v_gt, em_v_p, em_v_c):
#     ps.remove_all_structures()
#     ps.init()
#     ps.set_ground_plane_mode("shadow_only")
#     space = 100  # Spacing between meshes
#     # Convert tensors to numpy
#     vertices_gt = x2[0].cpu().numpy() + [0, 0, 0]
#     vertices_pred = x2[0].cpu().numpy() + [space, 0, 0]
#     vertices_computed = x2[0].cpu().numpy() + [space * 2, 0, 0]
#     # Function values
#     em_v_gt_np = em_v_gt.cpu().numpy()
#     em_v_p_np = em_v_p.cpu().numpy()
#     em_v_c_np = em_v_c.cpu().numpy()
#
#     threshold = em_v_p_np.mean()  # 0.008
#     em_v_gt_np[em_v_gt_np > threshold] = threshold
#     em_v_p_np[em_v_p_np > threshold] = threshold
#     em_v_c_np[em_v_c_np > threshold] = threshold
#     # Compute shared color range
#     vmin = min(em_v_gt_np.min(), em_v_p_np.min(), em_v_c_np.min())
#     vmax = max(em_v_gt_np.max(), em_v_p_np.max(), em_v_c_np.max())
#     # Register point clouds with shared colormap range
#     ps.register_point_cloud("GT", vertices_gt, enabled=True) \
#         .add_scalar_quantity("GT", em_v_gt_np, enabled=True, vminmax=(vmin, vmax), cmap='reds')
#     ps.register_point_cloud("Pred", vertices_pred, enabled=True) \
#         .add_scalar_quantity("Pred", em_v_p_np, enabled=True, vminmax=(vmin, vmax), cmap='reds')
#     ps.register_point_cloud("Computed", vertices_computed, enabled=True) \
#         .add_scalar_quantity("Computed", em_v_c_np, enabled=True, vminmax=(vmin, vmax), cmap='reds')
#     ps.show()
