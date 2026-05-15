# PGMUVI dtype policy

PGMUVI must not rely on PyTorch's global default floating dtype.

Default floating dtype for user-facing `Lightcurve` construction is `torch.float64`.

All newly created floating tensors must derive dtype and device from the relevant reference tensor,
normally `Lightcurve.xdata`, `train_x`, or another input tensor.

Use:

* `ref.new_tensor(value)`
* `ref.new_zeros(shape)`
* `ref.new_ones(shape)`
* `torch.as_tensor(value, dtype=ref.dtype, device=ref.device)`

Avoid:

* `torch.Tensor(...)`
* bare `torch.tensor(...)` for floating values
* `torch.as_tensor(..., dtype=torch.float32)`
* hard-coded `torch.float32`, `torch.float64`, or `.float()` unless there is a documented numerical
  reason
* relying on `torch.get_default_dtype()`

Integer, boolean, and index tensors should remain explicit:

* masks: `torch.bool`
* indices: `torch.long`

If a calculation intentionally uses a different internal dtype, it must:

1. explain why in a comment/docstring,

2. cast back to the owning object/input dtype before returning, unless the public API documents
   otherwise.
