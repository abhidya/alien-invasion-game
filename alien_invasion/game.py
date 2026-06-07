import argparse
from pathlib import Path

import alien_invasion.game_functions as gf
import matplotlib.pyplot as plt
import numpy as np
import pygame
import seaborn as sns
from alien_invasion.DQN import (
    DEFAULT_CHECKPOINT_DIR,
    ENEMY_ACTIONS,
    PILOT_ACTIONS,
    AlternatingSelfPlayTrainer,
    TrainingSnapshot,
    enemy_action_to_move,
    enemy_reward,
    one_hot,
    pilot_reward,
)
from alien_invasion.game_items import GameItems
from alien_invasion.game_stats import GameStats
from alien_invasion.settings import Settings
from tqdm import tqdm


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(
        x=np.asarray(array_counter),
        y=np.asarray(array_score),
        color="b",
        x_jitter=.1,
        line_kws={"color": "green"},
    )
    ax.set(xlabel="games", ylabel="score")
    plt.show()


def game_snapshot(stats: GameStats, game_items: GameItems) -> TrainingSnapshot:
    return TrainingSnapshot(
        score=int(stats.score),
        ships_left=int(stats.ships_left),
        aliens_left=len(game_items.aliens.sprites()),
    )


def active_role_for_episode(trainer: AlternatingSelfPlayTrainer, train_role: str, episode: int) -> str:
    if train_role == "alternate":
        return trainer.role_for_episode(episode)
    if train_role in {"pilot", "enemy"}:
        return train_role
    raise ValueError(f"Unknown train role {train_role!r}.")


def run_game(
    episodes=150,
    fps=1000,
    weights_path=None,
    enemy_weights_path=None,
    checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
    show_plot=True,
    train_role="alternate",
    alternate_every=1,
    seed=None,
):
    checkpoint_dir = Path(checkpoint_dir)

    pygame.init()
    fps_clock = pygame.time.Clock()
    ai_settings = Settings()

    trainer = AlternatingSelfPlayTrainer(
        checkpoint_dir=checkpoint_dir,
        pilot_weights_path=weights_path,
        enemy_weights_path=enemy_weights_path,
        alternate_every=alternate_every,
        seed=seed,
    )
    counter_games = 0
    score_plot = []
    counter_plot = []

    for episode in tqdm(range(1, episodes + 1)):
        role = active_role_for_episode(trainer, train_role, episode)
        pilot_training = role == "pilot"
        enemy_training = role == "enemy"

        stats = GameStats(ai_settings)
        game_items = GameItems(ai_settings, stats)
        gf.create_fleet(ai_settings, game_items)
        gf.start_new_game(ai_settings, stats, game_items)

        while stats.game_active:
            stats.time_passed = fps_clock.tick(fps) / 1000
            gf.check_events(ai_settings, stats, game_items)

            if stats.game_active:
                state_old = gf.get_state(ai_settings, stats, game_items)
                before = game_snapshot(stats, game_items)

                pilot_action = trainer.pilot.act(state_old, training=pilot_training)
                gf.do_move(one_hot(pilot_action, len(PILOT_ACTIONS)), ai_settings, stats, game_items)
                game_items.ship.update(stats)
                gf.update_bullets(ai_settings, stats, game_items)

                enemy_action = trainer.enemy.act(state_old, training=enemy_training)
                gf.update_aliens(
                    ai_settings,
                    stats,
                    game_items,
                    fleet_move=enemy_action_to_move(enemy_action),
                )

                state_new = gf.get_state(ai_settings, stats, game_items)
                after = game_snapshot(stats, game_items)
                done = not stats.game_active

                if pilot_training:
                    reward = pilot_reward(before, after, done)
                    trainer.pilot.train_short_memory(state_old, pilot_action, reward, state_new, done)
                if enemy_training:
                    reward = enemy_reward(before, after, done)
                    trainer.enemy.train_short_memory(state_old, enemy_action, reward, state_new, done)

            gf.update_screen(ai_settings, stats, game_items)

        active_agent = trainer.agent_for_role(role)
        active_agent.replay_new()
        counter_games += 1
        checkpoint = trainer.checkpoint(role=role, episode=episode, score=stats.score)
        print(
            {
                "episode": episode,
                "trained": role,
                "score": stats.score,
                "epsilon": round(checkpoint.epsilon, 4),
                "loss": active_agent.last_loss,
                "checkpoint": str(checkpoint.weights_path),
            }
        )
        score_plot.append(stats.score)
        counter_plot.append(counter_games)

    if show_plot:
        plot_seaborn(counter_plot, score_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train alternating pilot/enemy DQN agents.")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--fps", type=int, default=1000)
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional pilot weights path. Defaults to checkpoints/pilot.weights.h5.",
    )
    parser.add_argument(
        "--enemy-weights",
        default=None,
        help="Optional enemy weights path. Defaults to checkpoints/enemy.weights.h5.",
    )
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--train-role", choices=["alternate", "pilot", "enemy"], default="alternate")
    parser.add_argument("--alternate-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true", help="Skip the seaborn/matplotlib score plot.")
    args = parser.parse_args()
    run_game(
        episodes=args.episodes,
        fps=args.fps,
        weights_path=args.weights,
        enemy_weights_path=args.enemy_weights,
        checkpoint_dir=args.checkpoint_dir,
        show_plot=not args.no_plot,
        train_role=args.train_role,
        alternate_every=args.alternate_every,
        seed=args.seed,
    )
