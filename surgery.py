import flax.serialization
import jax.numpy as jnp
from jax import tree_util
import numpy as np

def inspect_msgpack(filepath):
    with open(filepath, "rb") as f:
        bytes_data = f.read()
    state_dict = flax.serialization.msgpack_restore(bytes_data)
    print("Keys in state_dict:")
    def print_shapes(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                print_shapes(v, prefix + k + "/")
            else:
                print(f"{prefix}{k}: {v.shape}")
    print_shapes(state_dict)

inspect_msgpack("modelo_treinado.msgpack")
