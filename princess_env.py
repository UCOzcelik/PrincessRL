# princess_env.py
import random


class PrincessEnv:
    """
    Grid-World mit:
    - Grid 12x7
    - Wasser oben/unten (unpassierbar)
    - Rotes Schloss links (Start/Ziel)
    - Prinzessin rechts, leicht randomisiert
    - Wände: Basis-Layout + 0–2 zufällige Extra-Wände
    - Gegner: Basis-Positionen, pro Episode leicht nach oben/unten verschoben
    - Actions: 0=hoch, 1=runter, 2=links, 3=rechts

    State = (x, y, has_princess) tabellarisch kodiert.
    """

    def __init__(self, width=12, height=7, max_steps=250):
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Wasser-Reihen
        self.water_rows = {0, height - 1}

        # Start (rotes Schloss)
        self.home_pos = (0, height // 2)

        # mögliche Princess-Positionen (kleine Variation rechts)
        mid = height // 2
        self.princess_spawn_positions = [
            (width - 1, mid),        # rechts Mitte
            (width - 1, mid - 1),    # eine Reihe höher
            (width - 1, mid + 1),    # eine Reihe tiefer
            (width - 2, mid),        # eine Spalte weiter links
        ]

        # ---------- Basis-Wände (dein schönes Layout) ----------
        self.base_walls = {
            (4, 2), (4, 3), (4, 4),       # vertikale Wand in der Mitte
            (7, 1), (7, 2),               # Block oben rechts
            (8, 4), (8, 5),               # Block unten rechts
        }

        # mögliche Extra-Wände, aus denen wir pro Episode 0–2 wählen
        self.extra_wall_candidates = [
            (2, 2), (2, 4),
            (6, 3),
            (9, 2), (9, 4),
        ]

        # ---------- Basis-Feinde ----------
        self.base_enemies = [
            (3, 2), (3, 4),
            (6, 2), (6, 4),
            (9, 3),
        ]

        # Tabellarischer State: (x, y, has_princess) -> int
        self.n_positions = self.width * self.height
        self.n_states = self.n_positions * 2  # has_princess = 0 oder 1
        self.n_actions = 4

        self.reset()

    # ---------- Helper ----------

    def _pos_to_index(self, x, y):
        return y * self.width + x

    def _state_to_index(self, x, y, has_princess):
        pos_idx = self._pos_to_index(x, y)
        return pos_idx + has_princess * self.n_positions

    def _index_to_state(self, state_idx):
        has_princess = 1 if state_idx >= self.n_positions else 0
        pos_idx = state_idx % self.n_positions
        y = pos_idx // self.width
        x = pos_idx % self.width
        return x, y, has_princess

    def _manhattan(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    # ---------- RL-API ----------

    def reset(self):
        """
        Neue Episode:
        - Prinzessin: zufällige Position rechts
        - Wände: Basis + 0–2 Extra-Wände
        - Gegner: von Basis-Positionen leicht nach oben/unten verschoben
        - Agent: am Schloss
        """
        # Prinzessin
        self.princess_pos = random.choice(self.princess_spawn_positions)

        # Wände
        walls = set(self.base_walls)
        max_extra = min(2, len(self.extra_wall_candidates))
        num_extra = random.randint(0, max_extra)
        extras = random.sample(self.extra_wall_candidates, k=num_extra)
        for (wx, wy) in extras:
            if (wx, wy) == self.home_pos:
                continue
            if (wx, wy) in self.princess_spawn_positions:
                continue
            if wy in self.water_rows:
                continue
            walls.add((wx, wy))
        self.wall_positions = walls

        # Gegner leicht variieren
        enemies = []
        for (ex, ey) in self.base_enemies:
            shift = random.choice([-1, 0, 1])  # hoch, gleich, runter
            new_y = ey + shift
            new_x = ex

            # gültig & kein Wasser & keine Wand
            if (
                0 < new_y < self.height - 1
                and (new_x, new_y) not in self.wall_positions
                and new_y not in self.water_rows
            ):
                pos = (new_x, new_y)
            else:
                pos = (ex, ey)

            # nicht direkt auf Start oder Princess-Spawn
            if pos == self.home_pos or pos in self.princess_spawn_positions:
                pos = (ex, ey)

            enemies.append(pos)

        # Doppelte raus
        self.enemy_positions = list(dict.fromkeys(enemies))

        # Agent reset
        self.agent_x, self.agent_y = self.home_pos
        self.has_princess = 0
        self.steps = 0

        return self._state_to_index(self.agent_x, self.agent_y, self.has_princess)

    def step(self, action):
        """
        Action: 0=hoch, 1=runter, 2=links, 3=rechts
        Rückgabe: next_state_idx, reward, done, info
        """
        self.steps += 1
        done = False
        info = {}

        # --- Basis-Reward: leichter Schritt-Preis ---
        reward = -0.5

        # alten Abstand zum Ziel merken (für Reward-Shaping)
        if self.has_princess == 0:
            tx, ty = self.princess_pos
        else:
            tx, ty = self.home_pos
        old_dist = self._manhattan(self.agent_x, self.agent_y, tx, ty)

        # Bewegung
        dx, dy = 0, 0
        if action == 0:
            dy = -1
        elif action == 1:
            dy = 1
        elif action == 2:
            dx = -1
        elif action == 3:
            dx = 1

        new_x = self.agent_x + dx
        new_y = self.agent_y + dy

        moved = False

        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            if new_y in self.water_rows or (new_x, new_y) in self.wall_positions:
                # gegen Wand / Wasser
                reward -= 2.0
            else:
                self.agent_x = new_x
                self.agent_y = new_y
                moved = True
        else:
            # aus dem Grid laufen
            reward -= 2.0

        # Feind getroffen?
        if (self.agent_x, self.agent_y) in self.enemy_positions:
            reward -= 40.0   # etwas weniger hart als -80
            done = True
            info["dead"] = True
            info["reason"] = "enemy"

        else:
            # Prinzessin einsammeln?
            if (
                self.has_princess == 0
                and (self.agent_x, self.agent_y) == self.princess_pos
            ):
                self.has_princess = 1
                reward += 40.0
                info["picked_princess"] = True

            # Mit Prinzessin zurück ins Schloss?
            if self.has_princess == 1 and (self.agent_x, self.agent_y) == self.home_pos:
                reward += 120.0
                done = True
                info["success"] = True

            # Reward-Shaping: Bewegung Richtung Ziel belohnen / weg bestrafen
            if moved and not done:
                if self.has_princess == 0:
                    tx, ty = self.princess_pos
                else:
                    tx, ty = self.home_pos
                new_dist = self._manhattan(self.agent_x, self.agent_y, tx, ty)
                if new_dist < old_dist:
                    reward += 0.6    # näher am Ziel → Bonus
                elif new_dist > old_dist:
                    reward -= 0.6    # weiter weg → Malus

        # Timeout
        if self.steps >= self.max_steps and not done:
            done = True
            info["timeout"] = True

        next_state_idx = self._state_to_index(
            self.agent_x, self.agent_y, self.has_princess
        )
        return next_state_idx, reward, done, info

    def render(self):
        """ASCII-Ansicht im Terminal zu Debugzwecken."""
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Wasser
        for y in self.water_rows:
            for x in range(self.width):
                grid[y][x] = "~"

        # Land
        for y in range(self.height):
            if y in self.water_rows:
                continue
            for x in range(self.width):
                grid[y][x] = "."

        # Wände
        for (wx, wy) in self.wall_positions:
            grid[wy][wx] = "#"

        # Schlösser & Prinzessin
        hx, hy = self.home_pos
        px, py = self.princess_pos
        grid[hy][hx] = "H"
        grid[py][px] = "P"

        # Feinde
        for (ex, ey) in self.enemy_positions:
            if (ex, ey) != (self.agent_x, self_agent_y := self.agent_y):
                grid[ey][ex] = "E"

        # Agent
        symbol = "A" if self.has_princess == 0 else "B"
        grid[self.agent_y][self.agent_x] = symbol

        print()
        for row in grid:
            print(" ".join(row))
        print()
