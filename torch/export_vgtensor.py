from torch import Tensor


def export_vgtensor(tensor: Tensor, filename: str):
    """Saves the numpy .tobytes() dump of a tensor."""

    with open(filename, "wb") as f:
        f.write(tensor.numpy().tobytes())
