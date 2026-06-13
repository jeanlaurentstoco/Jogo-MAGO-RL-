import flax
import jax
import jax.numpy as jnp
# Test logic for padding
old_param = jnp.ones((3, 3, 5, 32))
new_param = jnp.zeros((3, 3, 25, 32))
# the old param should be placed at the end? Or repeated?
# If old was 1 frame, new is 5 frames. The current frame is typically at the END of the channel list.
print("Done")
