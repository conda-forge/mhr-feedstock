import torch

from mhr.io import get_default_asset_folder, get_mhr_fbx_path
from mhr.mhr import MHRPoseCorrectivesModel
from mhr.utils import SparseLinear, batch6DFromXYZ


torch.manual_seed(0)

rotations = torch.zeros(2, 4, 3)
rotations_6d = batch6DFromXYZ(rotations)
assert rotations_6d.shape == (2, 4, 6)
assert torch.allclose(rotations_6d[..., 0], torch.ones(2, 4))
assert torch.allclose(rotations_6d[..., 4], torch.ones(2, 4))

joint_parameters = torch.zeros(2, 4, 7)
pose_predictor = torch.nn.Linear(12, 6, bias=False)
with torch.no_grad():
    pose_predictor.weight.fill_(0.5)
pose_model = MHRPoseCorrectivesModel(pose_predictor)
pose_offsets = pose_model(joint_parameters)
assert pose_offsets.shape == (2, 2, 3)
assert torch.count_nonzero(pose_offsets) == 0

sparse_mask = torch.tensor(
    [[True, False, True], [False, True, False]],
    dtype=torch.bool,
)
layer = SparseLinear(3, 2, sparse_mask, bias=False)
with torch.no_grad():
    layer.sparse_weight.fill_(1)
values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
expected = torch.tensor([[4.0, 2.0], [10.0, 5.0]])
assert torch.allclose(layer(values), expected)

asset_folder = get_default_asset_folder()
assert get_mhr_fbx_path(asset_folder, 2).endswith("assets/lod2.fbx")
