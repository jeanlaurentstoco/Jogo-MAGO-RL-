import numpy as np
enemy_pos = np.array([10, 10])
player_pos = np.array([10, 5])
to_enemy = enemy_pos - player_pos # [0, 5]
# Player presses 'w' (up) -> move = [0, -1]. Dot = -5 (fleeing!)
# Player presses 's' (down) -> move = [0, 1]. Dot = 5 (attacking!)
print("Up:", np.dot(np.array([0, -1]), to_enemy))
print("Down:", np.dot(np.array([0, 1]), to_enemy))
