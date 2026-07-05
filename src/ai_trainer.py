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

    # Train generation by generation so we can show replay
    for gen in range(1, config.TRAIN_GENERATIONS + 1):

        # 1. Show info screen (non-blocking)
        if not _draw_info(screen, gen, config.TRAIN_GENERATIONS,
                          best_fitness_overall):
            break  # user quit

        # 2. Evaluate one generation
        population.run(_eval_genomes, 1)

        # 3. Check best genome
        current_best = population.best_genome
        if current_best and current_best.fitness > best_fitness_overall:
            best_fitness_overall = current_best.fitness
            best_overall = current_best
            save_genome(current_best, 'best_genome.pkl')

        # 4. Replay best of this generation
        if current_best:
            if not replay_genome(screen, current_best, neat_config,
                                 gen_label=f'Gen {gen}'):
                break

    # Save final best
    if best_overall:
        path = save_genome(best_overall, 'best_genome.pkl')
        print(f'\nBest genome saved to {path}')
        print(f'Best fitness: {best_fitness_overall:.0f}')
    else:
        print('\nNo best genome found.')

    return best_overall
