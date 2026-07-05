"""AI Flappy Bird — Entry point.

Usage:
    python main.py            # Manual play
    python main.py --train    # Train AI (Phase 2)
    python main.py --replay   # Replay best AI (Phase 3)
"""

import sys
import pygame
import os
from src.game import FlappyBirdGame
from src import config
from src import ai_trainer


def draw_text(surface, text, size, x, y, color=config.WHITE, center=True):
    """Helper to draw text on surface."""
    font = pygame.font.SysFont('Arial', size)
    img = font.render(text, True, color)
    rect = img.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    surface.blit(img, rect)


def show_menu(screen):
    """Display the main menu and return the selected mode."""
    clock = pygame.time.Clock()
    selecting = True

    while selecting:
        clock.tick(config.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return 'manual'
                elif event.key == pygame.K_2:
                    return 'train'
                elif event.key == pygame.K_3:
                    return 'replay'
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # Draw
        screen.fill(config.BLUE_SKY)

        # Title
        draw_text(screen, '🐦 AI Flappy Bird', 64,
                  config.WINDOW_WIDTH // 2, 150, config.WHITE)
        draw_text(screen, '用 NEAT 遗传算法训练 AI 玩 Flappy Bird', 24,
                  config.WINDOW_WIDTH // 2, 220, config.WHITE)

        # Menu options
        draw_text(screen, '[1]  手动模式 — 你自己玩', 36,
                  config.WINDOW_WIDTH // 2, 320, config.YELLOW)
        draw_text(screen, '[2]  AI 训练模式', 36,
                  config.WINDOW_WIDTH // 2, 370, config.YELLOW)
        draw_text(screen, '[3]  回放最佳 AI', 36,
                  config.WINDOW_WIDTH // 2, 420, config.YELLOW)

        draw_text(screen, '按 ESC 退出', 20,
                  config.WINDOW_WIDTH // 2, 500, config.WHITE)

        pygame.display.flip()

    return None


def run_manual(screen):
    """Run the game in manual (keyboard) mode."""
    clock = pygame.time.Clock()
    game = FlappyBirdGame()
    running = True

    while running:
        clock.tick(config.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.step(jump=True)
                elif event.key == pygame.K_r and not game.alive:
                    game.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game.alive:
                    game.step(jump=True)
                else:
                    game.reset()

        # If bird is alive and no space was pressed, still step
        if game.alive:
            # Check if space is held down
            keys = pygame.key.get_pressed()
            game.step(jump=keys[pygame.K_SPACE])

        # Draw
        game.draw(screen)

        # Game over overlay
        if not game.alive:
            draw_text(screen, 'GAME OVER', 64,
                      config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2 - 40,
                      config.RED)
            draw_text(screen, f'得分: {game.score}', 36,
                      config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2 + 30,
                      config.WHITE)
            draw_text(screen, '按 R 重新开始  |  按 ESC 返回菜单', 24,
                      config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2 + 80,
                      config.WHITE)

        pygame.display.flip()


def main():
    """Main entry point."""
    # Parse command line args
    args = sys.argv[1:]

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption('AI Flappy Bird 🐦')

    if '--train' in args:
        ai_trainer.train(screen)
    elif '--replay' in args:
        genome = ai_trainer.load_genome('best_genome.pkl')
        if genome is None:
            print('[!] No saved model found. Train one first: python main.py --train')
            pygame.quit()
            return
        neat_config = ai_trainer.load_neat_config()
        if neat_config:
            ai_trainer.replay_genome(screen, genome, neat_config, gen_label='Best AI')
    else:
        # Show menu
        mode = show_menu(screen)
        if mode == 'manual':
            run_manual(screen)
        elif mode == 'train':
            ai_trainer.train(screen)
        elif mode == 'replay':
            genome = ai_trainer.load_genome('best_genome.pkl')
            if genome is None:
                draw_text(screen, 'No saved model found!', 36,
                          config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2,
                          config.RED)
                draw_text(screen, 'Train one first: python main.py --train', 24,
                          config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2 + 50,
                          config.WHITE)
                pygame.display.flip()
                pygame.time.wait(2000)
            else:
                neat_config = ai_trainer.load_neat_config()
                if neat_config:
                    ai_trainer.replay_genome(screen, genome, neat_config,
                                             gen_label='Best AI')

    pygame.quit()


if __name__ == '__main__':
    main()
