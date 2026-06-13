import flax.serialization
import jax.numpy as jnp
from jax import tree_util
import numpy as np

def convert_checkpoint(old_path, new_path):
    print(f"Lendo checkpoint antigo: {old_path}")
    with open(old_path, "rb") as f:
        bytes_data = f.read()
    old_state = flax.serialization.msgpack_restore(bytes_data)
    
    new_state = {"params": {}}
    old_p = old_state["params"]
    new_p = new_state["params"]
    
    # Mapping old unnamed to new named
    try:
        new_p["conv1"] = old_p["Conv_0"]
        new_p["res1"] = old_p["ResNetBlock_0"]
        new_p["conv2"] = old_p["Conv_1"]
        new_p["res2"] = old_p["ResNetBlock_1"]
        new_p["conv3"] = old_p["Conv_2"]
        new_p["res3"] = old_p["ResNetBlock_2"]
        
        new_p["dense_cnn"] = old_p["Dense_0"]
        new_p["dense_scalar"] = old_p["Dense_1"]
        new_p["dense_fusion"] = old_p["Dense_2"]
        
        new_p["memory"] = old_p["EpisodicMemory_0"]
        
        new_p["dense_h1"] = old_p["Dense_3"]
        new_p["dense_h2"] = old_p["Dense_4"]
        
        new_p["move_out"] = old_p["Dense_5"]
        new_p["shoot_out"] = old_p["Dense_6"]
        new_p["fatal_out"] = old_p["Dense_7"]
        new_p["dash_out"] = old_p["Dense_8"]
        new_p["wall_out"] = old_p["Dense_9"]
        new_p["aoe_out"] = old_p["Dense_10"]
        new_p["shield_out"] = old_p["Dense_11"]
        new_p["aim_out"] = old_p["Dense_12"]
        new_p["value_out"] = old_p["Dense_13"]
        
        new_p["aim_log_std"] = old_p["aim_log_std"]
        
        print("Mapeamento concluído com sucesso!")
        
        bytes_output = flax.serialization.to_bytes(new_state)
        with open(new_path, "wb") as f:
            f.write(bytes_output)
        print(f"Novo checkpoint salvo em: {new_path}")
    except KeyError as e:
        print(f"Erro ao mapear: Chave {e} não encontrada no checkpoint antigo.")

if __name__ == "__main__":
    convert_checkpoint("modelo_treinado.msgpack", "modelo_treinado_named.msgpack")
