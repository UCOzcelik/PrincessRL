# visual_demo.py
import numpy as np
import pygame
from princess_env import PrincessEnv

CELL_SIZE = 80
MARGIN = 2

COLOR_BG = (15, 25, 15)
COLOR_GRASS = (46, 100, 46)
COLOR_GRASS_DARK = (30, 70, 30)
COLOR_GRID = (10, 30, 10)
COLOR_WATER = (20, 40, 80)
COLOR_WALL = (60, 60, 60)
COLOR_SHADOW = (0, 0, 0, 120)


def load_sprite_bgcolor(path, size, color_tolerance=10):
    """Sprite laden, skalieren und einfarbigen Hintergrund (Pixel 0,0) transparent machen."""
    try:
        img = pygame.image.load(path).convert_alpha()
    except Exception as e:
        print(f"[!] Fehler beim Laden von {path}: {e}")
        return None

    if size is not None:
        img = pygame.transform.smoothscale(img, (size, size))

    bg_r, bg_g, bg_b, bg_a = img.get_at((0, 0))
    w, h = img.get_size()

    for x in range(w):
        for y in range(h):
            r, g, b, a = img.get_at((x, y))
            if a == 0:
                continue
            if (abs(r - bg_r) <= color_tolerance and
                abs(g - bg_g) <= color_tolerance and
                abs(b - bg_b) <= color_tolerance):
                img.set_at((x, y), (r, g, b, 0))
    return img


def load_sprite_checkerboard(path, size,
                             brightness_threshold=220,
                             saturation_threshold=25):
    """PNG-Kästchen-Hintergrund entfernen."""
    try:
        img = pygame.image.load(path).convert_alpha()
    except Exception as e:
        print(f"[!] Fehler beim Laden von {path}: {e}")
        return None

    if size is not None:
        img = pygame.transform.smoothscale(img, (size, size))

    w, h = img.get_size()
    for x in range(w):
        for y in range(h):
            r, g, b, a = img.get_at((x, y))
            if a == 0:
                continue
            brightness = (r + g + b) / 3.0
            color_diff = max(r, g, b) - min(r, g, b)
            if brightness > brightness_threshold and color_diff < saturation_threshold:
                img.set_at((x, y), (r, g, b, 0))
    return img


def draw_shadow(screen, x, y, size):
    shadow_w = int(size * 0.8)
    shadow_h = int(size * 0.3)
    surf = pygame.Surface((shadow_w, shadow_h), pygame.SRCALPHA)
    pygame.draw.ellipse(surf, COLOR_SHADOW, (0, 0, shadow_w, shadow_h))
    screen.blit(
        surf,
        (x + (size - shadow_w) // 2, y + size - shadow_h // 2)
    )


def draw_grid(screen, env, red_knight, red_princess_small, red_castle,
              blue_castle, blue_knight):
    tile = CELL_SIZE - 2 * MARGIN
    screen.fill(COLOR_BG)

    hx, hy = env.home_pos
    px, py = env.princess_pos

    # Hintergrund-Tiles
    for y in range(env.height):
        for x in range(env.width):
            rect = pygame.Rect(
                x * CELL_SIZE + MARGIN,
                y * CELL_SIZE + MARGIN,
                tile,
                tile
            )

            if y in env.water_rows:
                base = COLOR_WATER
            elif (x, y) in getattr(env, "wall_positions", set()):
                base = COLOR_WALL
            else:
                dist = abs(x - hx) + abs(y - hy)
                if dist == 0:
                    base = COLOR_GRASS_DARK
                elif dist == 1:
                    base = COLOR_GRASS
                else:
                    base = COLOR_GRASS

            pygame.draw.rect(screen, base, rect)
            pygame.draw.rect(screen, COLOR_GRID, rect, 1)

    def pos_to_px(cx, cy):
        return cx * CELL_SIZE + MARGIN, cy * CELL_SIZE + MARGIN

    # Rotes Schloss
    sx, sy = pos_to_px(*env.home_pos)
    draw_shadow(screen, sx, sy, tile)
    if red_castle:
        screen.blit(red_castle, (sx, sy))

    # Blaues Schloss (da wo die Prinzessin-Position ist)
    bx, by = pos_to_px(*env.princess_pos)
    draw_shadow(screen, bx, by, tile)
    if blue_castle:
        screen.blit(blue_castle, (bx, by))

    # Blaue Ritter (gespiegelt nach links)
    for (ex, ey) in env.enemy_positions:
        ex_px, ey_px = pos_to_px(ex, ey)
        draw_shadow(screen, ex_px, ey_px, tile)
        if blue_knight:
            screen.blit(blue_knight, (ex_px, ey_px))

    # Prinzessin (wenn nicht getragen) – klein und etwas über dem Turm
    if env.has_princess == 0 and red_princess_small:
        px_px, py_px = pos_to_px(px, py)
        draw_shadow(screen, px_px, py_px, tile)
        princess_y = py_px - int(tile * 0.25)
        # leicht zentrieren
        offset_x = (tile - red_princess_small.get_width()) // 2
        screen.blit(red_princess_small, (px_px + offset_x, princess_y))

    # Roter Ritter
    ax_px, ay_px = pos_to_px(env.agent_x, env.agent_y)
    draw_shadow(screen, ax_px, ay_px, tile)
    if red_knight:
        screen.blit(red_knight, (ax_px, ay_px))

    # Kleine Prinzessin beim Ritter (noch kleiner)
    if env.has_princess == 1 and red_princess_small:
        tiny_w = red_princess_small.get_width() // 2
        tiny_h = red_princess_small.get_height() // 2
        tiny = pygame.transform.smoothscale(red_princess_small, (tiny_w, tiny_h))
        screen.blit(tiny, (ax_px + tile // 2, ay_px))

    pygame.display.flip()


def run_visual_episode():
    # Q-Tabelle laden
    Q = np.load("q_table.npy")
    env = PrincessEnv()

    pygame.init()
    pygame.display.set_caption("Princess RL – Random Map Demo")

    screen = pygame.display.set_mode(
        (env.width * CELL_SIZE, env.height * CELL_SIZE)
    )
    clock = pygame.time.Clock()

    tile_size = CELL_SIZE - 2 * MARGIN

    red_knight = load_sprite_bgcolor("redknight.png", tile_size)
    blue_knight = load_sprite_bgcolor("blueknight.png", tile_size)
    red_castle = load_sprite_bgcolor("redcastle.png", tile_size)
    blue_castle = load_sprite_bgcolor("bluecastle.png", tile_size)

    # Prinzessin bewusst kleiner laden (z.B. 65% vom Tile)
    princess_size = int(tile_size * 0.65)
    red_princess_small = load_sprite_checkerboard("redprincess.png", princess_size)

    # Blaue Ritter spiegeln, damit sie nach links schauen
    if blue_knight:
        blue_knight = pygame.transform.flip(blue_knight, True, False)

    state = env.reset()
    done = False
    total_reward = 0.0
    last_info = {}

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                done = True

        if not done:
            action = int(np.argmax(Q[state]))
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            last_info = info

        draw_grid(screen, env, red_knight, red_princess_small,
                  red_castle, blue_castle, blue_knight)

        if done:
            success = bool(last_info.get("success", False))
            print(f"Episode beendet – Reward: {total_reward:.1f}, success={success}")
            # warten, bis Fenster geschlossen wird
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                clock.tick(30)
            running = False
        else:
            clock.tick(2)

    pygame.quit()


if __name__ == "__main__":
    run_visual_episode()
