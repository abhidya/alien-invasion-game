import pygame
from DQN import DQNAgent
import game_functions as gf
from game_items import GameItems
from game_stats import GameStats
from settings import Settings
from random import randint

# FPS = 60


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
    counter_plot =[]
    record = 0

    # FOR THE DQN #

    while counter_games < 150:

        # Create statistics.
        stats = GameStats(ai_settings)

        # Create game items.
        game_items = GameItems(ai_settings, stats)

        # Create a fleet of aliens.
        gf.create_fleet(ai_settings, game_items)

        # Start the main loop for the game.
        while True:

            # FOR THE DQN #

            agent.epsilon = 80 - counter_games
            #state_old = agent.get_state(game, player1, food1)

            #TO:DO  if randint(0, 200) < agent.epsilon:
            # TO:DO    final_move = to_categorical(randint(0, 2), num_classes=3)
            #TO:DO            else:
                # predict action based on the old state
                # TO:DO   prediction = agent.model.predict(state_old.reshape((1,11)))
                # TO:DO   final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

            # FOR THE DQN #



            stats.time_passed = fps_clock.tick(FPS) / 1000  # Time in seconds since previous loop.

            gf.check_events(ai_settings, stats, game_items)

            if stats.game_active:
                game_items.ship.update(stats)
                gf.update_bullets(ai_settings, stats, game_items)
                gf.update_aliens(ai_settings, stats, game_items)



                # DQN #

                # perform new move and get new state
                # TO:DO  do_move(final_move, game, agent)
                # TO:DO  state_new = agent.get_state(game, player1, food1)

                # set treward for the new state
                # TO:DO reward = agent.set_reward(player1, game.crash)

                # train short memory base on the new action and state
                # TO:DO agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)

                # store the new data into a long term memory
                # TO:DO  agent.remember(state_old, final_move, reward, state_new, game.crash)
                # Get value of played game
                # TO:DO record = get_record(game.score, record)
                # DQN #


            gf.update_screen(ai_settings, stats, game_items)

        # FOR THE DQN #
        #TO:DO agent.replay_new(agent.memory)
        #TO:DOcounter_games += 1
        #TO:DOprint('Game', counter_games, '      Score:', game.score)
        #TO:DO score_plot.append(game.score)
        #TO:DO counter_plot.append(counter_games)
        # FOR THE DQN #


if __name__ == '__main__':
    run_game()
