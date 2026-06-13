import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util

class LightModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.Dense(64)(x)
        return x

class HeavyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.Conv(32, (3, 3))(x) # Extra layer
        x = nn.Dense(64)(x)
        return x

rng = jax.random.PRNGKey(0)
light = LightModel()
light_params = light.init(rng, jnp.ones((1, 10, 10, 5)))
bytes_data = flax.serialization.to_bytes(light_params)

heavy = HeavyModel()
heavy_params = heavy.init(rng, jnp.ones((1, 10, 10, 5)))

try:
    restored = flax.serialization.from_bytes(heavy_params, bytes_data)
    print("Sucesso!")
except Exception as e:
    print(f"Erro: {e}")
