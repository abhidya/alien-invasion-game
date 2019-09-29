import pygame

import game_functions as gf
from game_items import GameItems
from game_stats import GameStats
from settings import Settings
from score_db import connect_and_collect, add_score
from end_display import ask
# FPS = 60


def run_game():

    FPS = 60

    # Initialize game, settings and create a screen object.
    pygame.init()
    fps_clock = pygame.time.Clock()
    ai_settings = Settings()

    # Create statistics.
    stats = GameStats(ai_settings)

    # Create game items.
    game_items = GameItems(ai_settings, stats)

    # Create a fleet of aliens.
    gf.create_fleet(ai_settings, game_items)

    # Start the main loop for the game.
    played = False
    while True:
		
        stats.time_passed = fps_clock.tick(FPS) / 1000  # Time in seconds since previous loop.

        gf.check_events(ai_settings, stats, game_items)
		
        if stats.game_active:
            game_items.ship.update(stats)
            gf.update_bullets(ai_settings, stats, game_items)
            gf.update_aliens(ai_settings, stats, game_items)
            played = True
        elif played:
            user = ask(game_items.screen)
            if len(user) > 0:
                coll = connect_and_collect()
                add_score(user , stats.score, coll)
            played = False
        
        gf.update_screen(ai_settings, stats, game_items)



if __name__ == '__main__':
    run_game()
