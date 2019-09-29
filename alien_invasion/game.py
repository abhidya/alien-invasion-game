from random import randint

import alien_invasion.game_functions as gf
import matplotlib.pyplot as plt
import numpy as np
import pygame
import seaborn as sns
from alien_invasion.game_items import GameItems
from alien_invasion.game_stats import GameStats
from keras.utils import to_categorical
from alien_invasion.settings import Settings
from tqdm import tqdm
from alien_invasion.DQN import DQNAgent


# FPS = 60
def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1,
                     line_kws={'color': 'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()


def run_game():
    FPS = 60

    # Initialize game, settings and create a screen object.
    pygame.init()
    fps_clock = pygame.time.Clock()
    ai_settings = Settings()

    # FOR THE DQN #

    agent = DQNAgent()
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0

    # FOR THE DQN #

    for i in tqdm(range(1, 150)):

        # Create statistics.
        stats = GameStats(ai_settings)

        # Create game items.
        game_items = GameItems(ai_settings, stats)

        # Create a fleet of aliens.
        gf.create_fleet(ai_settings, game_items)
        played = False

        gf.start_new_game(ai_settings, stats, game_items)

        # Start the main loop for the game.
        while stats.game_active:
            stats.time_passed = fps_clock.tick(FPS) / 1000  # Time in seconds since previous loop.

            gf.check_events(ai_settings, stats, game_items)

            if stats.game_active:
                game_items.ship.update(stats)
                gf.update_bullets(ai_settings, stats, game_items)
                gf.update_aliens(ai_settings, stats, game_items)
                # FOR THE DQN #
                agent.epsilon = 80 - counter_games
                state_old = gf.get_state(ai_settings, stats, game_items)
                if randint(0, 200) < agent.epsilon:
                    final_move = to_categorical(randint(0, 3), num_classes=4)
                else:
                    # predict action based on the old state
                    prediction = agent.model.predict(state_old.reshape((1, 3536)))
                    final_move = to_categorical(np.argmax(prediction[0]), num_classes=4)
                    # played = True

                # FOR THE DQN #

                # DQN #
                # perform new move and get new state
                gf.do_move(final_move, ai_settings, stats, game_items)

                state_new = gf.get_state(ai_settings, stats, game_items)

                # set reward for the new state
                reward = agent.set_reward(stats.score, stats.ships_left)

                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, stats.game_active)

                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new,  stats.game_active)
                # DQN #


            # gf.update_screen(ai_settings, stats, game_items)

        # FOR THE DQN #
        agent.replay_new(agent.memory)
        counter_games += 1
        print('Game', counter_games, '      Score:', stats.score)
        score_plot.append(stats.score)
        counter_plot.append(counter_games)
    agent.model.save_weights('weights.hdf5')
    plot_seaborn(counter_plot, score_plot)
    # FOR THE DQN #


if __name__ == '__main__':
    run_game()
