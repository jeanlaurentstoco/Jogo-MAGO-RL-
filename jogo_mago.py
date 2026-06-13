from random import randint, shuffle
from collections import deque
import keyboard
# pyrefly: ignore [missing-import]
import numpy as np
import time

class Jogador:
    def __init__(self, nome, pos):
        self.nome = nome
        self.forca = randint(1, 10)
        self.vida = randint(50, 100)
        self.pos = pos
        self.alcance = randint(2, 6)

class Inimigo:
    def __init__(self, nome, pos):
        self.nome = nome
        self.forca = randint(1, 10)
        self.vida = randint(50, 100)
        self.pos = pos

class Game:
    def __init__(self, jogador_nome, inimigo_nome, tamanho_mapa):
        self.tamanho_mapa = tamanho_mapa
        self.mapa = np.zeros((self.tamanho_mapa, self.tamanho_mapa), dtype=int)
        pos_ini = [randint(1, self.tamanho_mapa // 2 - 1), randint(1, self.tamanho_mapa // 2 - 1)]
        pos_jog = [randint(self.tamanho_mapa // 2, self.tamanho_mapa - 2), randint(self.tamanho_mapa // 2, self.tamanho_mapa - 2)]
        self.jogador = Jogador(jogador_nome, pos_jog)
        self.inimigo = Inimigo(inimigo_nome, pos_ini)
        self.paredes = set()
        self.gerar_paredes()
        self.projeteis = []
        self.mensagem = ""

    def bfs_conectado(self, inicio, fim, paredes):
        visitado = set()
        fila = deque()
        fila.append((inicio[0], inicio[1]))
        visitado.add((inicio[0], inicio[1]))
        alvo = (fim[0], fim[1])
        while fila:
            x, y = fila.popleft()
            if (x, y) == alvo:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.tamanho_mapa and 0 <= ny < self.tamanho_mapa:
                    if (nx, ny) not in visitado and (nx, ny) not in paredes:
                        visitado.add((nx, ny))
                        fila.append((nx, ny))
        return False

    def gerar_paredes(self):
        self.paredes = set()
        pos_j = (self.jogador.pos[0], self.jogador.pos[1])
        pos_i = (self.inimigo.pos[0], self.inimigo.pos[1])
        candidatas = []
        for x in range(self.tamanho_mapa):
            for y in range(self.tamanho_mapa):
                if (x, y) != pos_j and (x, y) != pos_i:
                    candidatas.append((x, y))
        shuffle(candidatas)
        num_paredes = int(self.tamanho_mapa * self.tamanho_mapa * 0.10)
        for c in candidatas:
            if len(self.paredes) >= num_paredes:
                break
            self.paredes.add(c)
            if not self.bfs_conectado(list(pos_j), list(pos_i), self.paredes):
                self.paredes.remove(c)

    def andar(self):
        mover = False
        nova_pos = list(self.jogador.pos)
        if keyboard.is_pressed('w'):
            nova_pos[1] -= 1
            mover = True
        elif keyboard.is_pressed('s'):
            nova_pos[1] += 1
            mover = True
        elif keyboard.is_pressed('a'):
            nova_pos[0] -= 1
            mover = True
        elif keyboard.is_pressed('d'):
            nova_pos[0] += 1
            mover = True

        if mover and 0 <= nova_pos[0] < self.tamanho_mapa and 0 <= nova_pos[1] < self.tamanho_mapa:
            if (nova_pos[0], nova_pos[1]) not in self.paredes:
                self.jogador.pos = nova_pos

    def atacar(self):
        dx, dy = 0, 0
        if keyboard.is_pressed('up'):
            dy = -1
        if keyboard.is_pressed('down'):
            dy = 1
        if keyboard.is_pressed('left'):
            dx = -1
        if keyboard.is_pressed('right'):
            dx = 1

        if dx == 0 and dy == 0:
            return

        self.projeteis.append({
            "x": self.jogador.pos[0],
            "y": self.jogador.pos[1],
            "dx": dx,
            "dy": dy,
            "colisoes": 0,
            "max_colisoes": self.jogador.alcance,
            "rastro": []
        })

    def atualizar_projeteis(self):
        novos = []
        for p in self.projeteis:
            next_x = p["x"] + p["dx"]
            next_y = p["y"] + p["dy"]

            ricocheteou = False

            if next_x < 0 or next_x >= self.tamanho_mapa:
                p["dx"] = -p["dx"]
                ricocheteou = True
            if next_y < 0 or next_y >= self.tamanho_mapa:
                p["dy"] = -p["dy"]
                ricocheteou = True

            if not ricocheteou and (next_x, next_y) in self.paredes:
                parede_x = (p["x"] + p["dx"], p["y"]) in self.paredes
                parede_y = (p["x"], p["y"] + p["dy"]) in self.paredes

                if parede_x and parede_y:
                    p["dx"] = -p["dx"]
                    p["dy"] = -p["dy"]
                elif parede_x:
                    p["dx"] = -p["dx"]
                else:
                    p["dy"] = -p["dy"]
                ricocheteou = True

            if ricocheteou:
                p["colisoes"] += 1
                if p["colisoes"] > p["max_colisoes"]:
                    continue
                next_x = p["x"] + p["dx"]
                next_y = p["y"] + p["dy"]

            if not (0 <= next_x < self.tamanho_mapa and 0 <= next_y < self.tamanho_mapa):
                continue

            if (next_x, next_y) in self.paredes:
                continue

            p["x"] = next_x
            p["y"] = next_y
            p["rastro"].append([next_x, next_y])

            if [next_x, next_y] == self.inimigo.pos:
                self.inimigo.vida -= self.jogador.forca
                self.mensagem = f"{self.jogador.nome} atingiu {self.inimigo.nome}! -{self.jogador.forca} HP"
                if self.inimigo.vida <= 0:
                    self.inimigo.vida = 0
                    self.mensagem += f" {self.inimigo.nome} derrotado!"
                continue

            novos.append(p)
        self.projeteis = novos

    def montar_mapa(self):
        self.mapa[:] = 0
        for px, py in self.paredes:
            self.mapa[py, px] = 4
        if self.inimigo.vida > 0:
            self.mapa[self.inimigo.pos[1], self.inimigo.pos[0]] = 1
        self.mapa[self.jogador.pos[1], self.jogador.pos[0]] = 2
        for p in self.projeteis:
            for r in p["rastro"][-3:]:
                if 0 <= r[0] < self.tamanho_mapa and 0 <= r[1] < self.tamanho_mapa:
                    if self.mapa[r[1], r[0]] == 0:
                        self.mapa[r[1], r[0]] = 3

    def printar_mapa(self):
        print("\033[H", end="")
        for i in range(self.tamanho_mapa):
            linha = []
            for j in range(self.tamanho_mapa):
                val = self.mapa[i, j]
                if val == 1:
                    linha.append("\033[32mE\033[0m")
                elif val == 2:
                    linha.append("\033[31mP\033[0m")
                elif val == 3:
                    linha.append("\033[33m*\033[0m")
                elif val == 4:
                    linha.append("\033[37m#\033[0m")
                else:
                    linha.append(".")
            print(" ".join(linha))
        print(f"\n{self.jogador.nome}: HP={self.jogador.vida} ATK={self.jogador.forca} Bounces={self.jogador.alcance}")
        print(f"{self.inimigo.nome}: HP={self.inimigo.vida} ATK={self.inimigo.forca}")
        if self.mensagem:
            print(f"\033[33m{self.mensagem}\033[0m")
        else:
            print(" " * 60)

    def update(self):
        self.andar()
        self.atacar()
        self.atualizar_projeteis()
        self.montar_mapa()
        self.printar_mapa()

if __name__ == "__main__":
    print("\033[2J\033[H", end="")
    game = Game("Jogador", "Inimigo", 30)
    while True:
        game.update()
        if keyboard.is_pressed('esc'):
            break
        time.sleep(0.1)

#deixar tudo em matriz, adicionar poderes, treinar ia com jax 