"""NEAT AI trainer for Flappy Bird.

Handles:
- Training: evaluate genomes across generations
- Replay: play back a saved genome visually
"""

import os
import pickle
import neat
import pygame
from src.game import FlappyBirdGame
from src import config


# ─── Config path ────────────────────────────────────────────────

def _get_neat_config_path():
    """Return the absolute path to the NEAT config file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'neat_config.txt')


def load_neat_config():
    """Load and return a NEAT Config object.

    Returns:
        neat.Config, or None if the config file is missing/invalid.
    """
    try:
        return neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            _get_neat_config_path(),
        )
    except Exception as e:
        print(f'[!] Failed to load NEAT config: {e}')
        return None


# ─── Evaluation ─────────────────────────────────────────────────

def _eval_genomes(genomes, neat_config):
    """Evaluate fitness for a list of genomes (called by neat-python).

    Args:
        genomes: list of (genome_id, genome) tuples
        neat_config: NEAT configuration object
    """
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
        game = FlappyBirdGame()

        while game.alive and game.frame_count < config.NEAT_MAX_FRAMES:
            inputs = game.get_inputs()
            output = net.activate(inputs)
            # Output > 0.5 means jump
            game.step(jump=output[0] > 0.5)

        genome.fitness = game.get_fitness()


# ─── Replay ──────────────────────────────────────────────────────

def replay_genome(screen, genome, neat_config, gen_label=None):
    """Let a genome play the game visually.

    Args:
        screen: pygame display surface
        genome: NEAT genome to replay
        neat_config: NEAT configuration
        gen_label: optional string shown on screen (e.g. 'Gen 5')

    Returns:
        True if replay completed, False if user pressed ESC/quit.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    game = FlappyBirdGame()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 24)
    max_replay_frames = 3000  # avoid hanging on a perfect bird

    while game.alive and game.frame_count < max_replay_frames:
        clock.tick(config.FPS * 2)  # 2x speed replay

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                # Space toggles pause
                if event.key == pygame.K_SPACE:
                    paused = True
                    while paused:
                        for ev in pygame.event.get():
                            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                                paused = False
                            if ev.type == pygame.QUIT:
                                return False

        # AI decision
        inputs = game.get_inputs()
        output = net.activate(inputs)
        game.step(jump=output[0] > 0.5)

        # Render
        game.draw(screen)

        # Overlay: generation label
        if gen_label:
            label = font.render(gen_label, True, config.WHITE)
            screen.blit(label, (config.WINDOW_WIDTH - 150, 15))

        # Overlay: fitness / score
        info = font.render(
            f'Score: {game.score}  Fit: {game.get_fitness():.0f}',
            True, config.WHITE,
        )
        screen.blit(info, (config.WINDOW_WIDTH - 250, 45))

        pygame.display.flip()

    # Brief delay so user can see the final state
    pygame.time.wait(300)
    return True


# ─── Save / Load ────────────────────────────────────────────────

def save_genome(genome, filename='best_genome.pkl'):
    """Pickle a genome to models/ directory.

    Returns:
        Full path of the saved file.
    """
    os.makedirs('models', exist_ok=True)
    path = os.path.join('models', filename)
    with open(path, 'wb') as f:
        pickle.dump(genome, f)
    return path


def load_genome(filename='best_genome.pkl'):
    """Load a pickled genome from models/ directory.

    Returns:
        NEAT genome object, or None if file not found.
    """
    path = os.path.join('models', filename)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─── Training ────────────────────────────────────────────────────

def _draw_info(screen, gen, total_gen, best_fit):
    """Draw a 'Training in progress' overlay."""
    font = pygame.font.SysFont('Arial', 32)
    font_small = pygame.font.SysFont('Arial', 22)

    screen.fill(config.BLUE_SKY)

    lines = [
        ('Training AI ...', 48, config.WHITE, config.WINDOW_WIDTH // 2, 200),
        (f'Generation  {gen} / {total_gen}', 32, config.YELLOW,
         config.WINDOW_WIDTH // 2, 280),
        (f'Best fitness so far:  {best_fit:.0f}', 28, config.WHITE,
         config.WINDOW_WIDTH // 2, 340),
        ('(Replaying best bird after each generation)', 22, config.GRAY,
         config.WINDOW_WIDTH // 2, 400),
        ('ESC to stop early', 20, config.GRAY,
         config.WINDOW_WIDTH // 2, 460),
    ]

    for text, size, color, x, y in lines:
        f = pygame.font.SysFont('Arial', size)
        img = f.render(text, True, color)
        rect = img.get_rect(center=(x, y))
        screen.blit(img, rect)

    # Also handle events so window doesn't freeze
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False

    pygame.display.flip()
    return True


# ─── Fitness chart ──────────────────────────────────────────────

def _show_fitness_chart(screen, history):
    """Draw a fitness-over-generations chart using pygame.

    Args:
        screen: pygame display surface.
        history: list of (gen, max_fitness, avg_fitness) tuples.
    """
    if not history:
        return

    margin = 60
    chart_w = config.WINDOW_WIDTH - margin * 2
    chart_h = config.WINDOW_HEIGHT - margin * 2 - 60
    chart_x = margin
    chart_y = margin + 30

    gens = [h[0] for h in history]
    max_fits = [h[1] for h in history]
    avg_fits = [h[2] for h in history]
    all_vals = max_fits + avg_fits
    y_min = min(all_vals) if all_vals else 0
    y_max = max(all_vals) if all_vals else 1
    if y_max == y_min:
        y_max = y_min + 1
    y_range = y_max - y_min

    def to_screen(g, v):
        x = chart_x + (g - gens[0]) / max(gens[-1] - gens[0], 1) * chart_w
        y = chart_y + chart_h - (v - y_min) / y_range * chart_h
        return int(x), int(y)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18)
    font_title = pygame.font.SysFont('Arial', 28)

    waiting = True
    while waiting:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                waiting = False

        screen.fill(config.BLUE_SKY)

        # Title
        title = font_title.render('Training Progress', True, config.WHITE)
        screen.blit(title, (config.WINDOW_WIDTH // 2 - title.get_width() // 2, 20))

        # Chart background
        bg_rect = pygame.Rect(chart_x, chart_y, chart_w, chart_h)
        pygame.draw.rect(screen, config.BLACK, bg_rect, 1)
        pygame.draw.rect(screen, (20, 20, 40), bg_rect)

        # Grid lines (4 horizontal)
        for i in range(5):
            y = chart_y + chart_h // 4 * i
            pygame.draw.line(screen, (60, 60, 60), (chart_x, y),
                             (chart_x + chart_w, y), 1)
            val = y_max - (y_max - y_min) / 4 * i
            label = font.render(f'{val:.0f}', True, config.GRAY)
            screen.blit(label, (chart_x - 50, y - 8))

        # X axis label
        xlbl = font.render('Generation', True, config.GRAY)
        screen.blit(xlbl, (chart_x + chart_w // 2 - 40, chart_y + chart_h + 5))

        # Draw lines
        if len(gens) > 1:
            # Max fitness (yellow)
            for i in range(1, len(gens)):
                p1 = to_screen(gens[i - 1], max_fits[i - 1])
                p2 = to_screen(gens[i], max_fits[i])
                pygame.draw.line(screen, config.YELLOW, p1, p2, 2)

            # Avg fitness (cyan)
            for i in range(1, len(gens)):
                p1 = to_screen(gens[i - 1], avg_fits[i - 1])
                p2 = to_screen(gens[i], avg_fits[i])
                pygame.draw.line(screen, (0, 200, 255), p1, p2, 2)

        # Legend
        leg_y = chart_y + chart_h + 30
        pygame.draw.line(screen, config.YELLOW, (chart_x, leg_y),
                         (chart_x + 30, leg_y), 2)
        lbl = font.render('Best fitness', True, config.YELLOW)
        screen.blit(lbl, (chart_x + 35, leg_y - 8))

        pygame.draw.line(screen, (0, 200, 255),
                         (chart_x + 160, leg_y), (chart_x + 190, leg_y), 2)
        lbl = font.render('Average fitness', True, (0, 200, 255))
        screen.blit(lbl, (chart_x + 195, leg_y - 8))

        # Final stats
        final = font.render(
            f'Generations: {gens[-1]}  |  Best: {max_fits[-1]:.0f}  |  Avg: {avg_fits[-1]:.0f}',
            True, config.WHITE,
        )
        screen.blit(final, (chart_x, chart_y + chart_h + 60))

        hint = font.render('Press any key or click to close', True, config.GRAY)
        screen.blit(hint, (chart_x + chart_w - 220, chart_y + chart_h + 60))

        pygame.display.flip()


def train(screen):
    """Run NEAT training with per-generation visualisation.

    Args:
        screen: pygame display surface.

    Returns:
        The best genome found during training.
    """
    # Load NEAT config
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        _get_neat_config_path(),
    )

    # Setup population
    population = neat.Population(neat_config)

    # --- Reporters (text output) ---
    population.add_reporter(neat.StdOutReporter(True))
    stats_reporter = neat.StatisticsReporter()
    population.add_reporter(stats_reporter)

    best_overall = None
    best_fitness_overall = 0.0
    history = []  # list of (gen, max_fit, avg_fit)

    # Train generation by generation so we can show replay
    for gen in range(1, config.TRAIN_GENERATIONS + 1):

        # 1. Show info screen (non-blocking)
        if not _draw_info(screen, gen, config.TRAIN_GENERATIONS,
                          best_fitness_overall):
            break  # user quit

        # 2. Evaluate one generation
        population.run(_eval_genomes, 1)

        # 3. Record stats
        best_gen = population.best_genome
        if best_gen:
            fit = best_gen.fitness
        else:
            fit = 0

        # Calculate average fitness from species stats
        species = population.species.species
        total_fit = sum(s.fitness for s in species.values())
        avg_fit = total_fit / len(species) if species else 0
        history.append((gen, fit, avg_fit))

        # 4. Check best genome
        if best_gen and fit > best_fitness_overall:
            best_fitness_overall = fit
            best_overall = best_gen
            save_genome(best_gen, 'best_genome.pkl')

        # 5. Replay best of this generation
        if best_gen:
            if not replay_genome(screen, best_gen, neat_config,
                                 gen_label=f'Gen {gen}'):
                break

    # Save final best
    if best_overall:
        path = save_genome(best_overall, 'best_genome.pkl')
        print(f'\nBest genome saved to {path}')
        print(f'Best fitness: {best_fitness_overall:.0f}')
    else:
        print('\nNo best genome found.')

    # Show fitness chart
    if history:
        _show_fitness_chart(screen, history)

    return best_overall
