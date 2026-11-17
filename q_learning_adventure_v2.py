"""Q-Learning text adventure (v2: stochastic world, more items, baseline).

Map (Hall is the centre):

                Vault (dragon, locked)
                   |
 Swamp - Armory - Hall - Library - Garden
                   |
               Entrance

Items : sword (Armory), potion (Swamp), key (Library), coin (Garden)
Events: Swamp has a 30% snake-bite chance on entry (-10 reward).
        Dragon fight is random: sword+potion -> 90% win, sword only -> 60%,
        no sword -> always lose.
Goal  : enter the Vault with the key and beat the dragon.
"""
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOMS = ["Entrance", "Hall", "Armory", "Library", "Vault", "Swamp", "Garden"]
ACTIONS = ["north", "south", "east", "west", "pickup"]
MOVES = {
    0: {"north": 1},
    1: {"south": 0, "west": 2, "east": 3, "north": 4},
    2: {"east": 1, "west": 5},
    3: {"west": 1, "east": 6},
    5: {"east": 2},
    6: {"west": 3},
}
ITEM_IN_ROOM = {2: "sword", 5: "potion", 3: "key", 6: "coin"}
ITEM_REWARD = {"sword": 10, "potion": 10, "key": 10, "coin": 5}
ITEMS = ["sword", "potion", "key", "coin"]
MAX_STEPS = 40


class Adventure:
    def reset(self):
        self.room, self.steps = 0, 0
        self.inv = {i: 0 for i in ITEMS}
        return self.state()

    def state(self):
        return (self.room, *[self.inv[i] for i in ITEMS])

    def step(self, a):
        """Returns (next_state, reward, done, won)."""
        self.steps += 1
        reward, done, won = -1, False, False  # step penalty
        act = ACTIONS[a]
        if act == "pickup":
            item = ITEM_IN_ROOM.get(self.room)
            if item and not self.inv[item]:
                self.inv[item] = 1
                reward += ITEM_REWARD[item]
            else:
                reward -= 2  # nothing useful to pick up
        elif act in MOVES.get(self.room, {}):
            nxt = MOVES[self.room][act]
            if nxt == 4:  # vault
                if not self.inv["key"]:
                    reward -= 4  # door locked, stay in Hall
                else:
                    self.room = nxt
                    p_win = 0.9 if (self.inv["sword"] and self.inv["potion"]) \
                        else 0.6 if self.inv["sword"] else 0.0
                    done = True
                    if random.random() < p_win:
                        reward, won = 100, True
                    else:
                        reward = -50
            else:
                self.room = nxt
                if nxt == 5 and random.random() < 0.3:  # snake bite
                    reward -= 10
        else:
            reward -= 2  # wall / invalid move
        if self.steps >= MAX_STEPS:
            done = True
        return self.state(), reward, done, won


def s_idx(s):
    room, *flags = s
    idx = room
    for f in flags:
        idx = idx * 2 + f
    return idx


N_STATES = len(ROOMS) * 2 ** len(ITEMS)


def train(episodes=20000, alpha=0.1, gamma=0.95,
          eps_start=1.0, eps_end=0.05, seed=0):
    random.seed(seed); np.random.seed(seed)
    env = Adventure()
    Q = np.zeros((N_STATES, len(ACTIONS)))
    rewards, wins = [], []
    for ep in range(episodes):
        eps = max(eps_end, eps_start * (1 - ep / (0.7 * episodes)))
        s, total, done, won = env.reset(), 0, False, False
        while not done:
            i = s_idx(s)
            a = random.randrange(len(ACTIONS)) if random.random() < eps \
                else int(np.argmax(Q[i]))
            s2, r, done, won = env.step(a)
            # Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
            target = r + (0 if done else gamma * np.max(Q[s_idx(s2)]))
            Q[i, a] += alpha * (target - Q[i, a])
            s, total = s2, total + r
        rewards.append(total)
        wins.append(int(won))
    return Q, rewards, wins


def evaluate(policy, episodes=1000, seed=123):
    random.seed(seed)
    env, wins, totals = Adventure(), 0, []
    for _ in range(episodes):
        s, total, done = env.reset(), 0, False
        while not done:
            s, r, done, won = env.step(policy(s))
            total += r
        wins += won
        totals.append(total)
    return wins / episodes, float(np.mean(totals))


def plot_curves(rewards, wins, path="reward_curve.png", window=500):
    k = np.ones(window) / window
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(np.convolve(rewards, k, mode="valid"))
    ax[0].set(title=f"Reward per episode ({window}-ep moving avg)",
              xlabel="Training episode", ylabel="Cumulative reward")
    ax[1].plot(np.convolve(wins, k, mode="valid"))
    ax[1].set(title=f"Win rate during training ({window}-ep moving avg)",
              xlabel="Training episode", ylabel="Win rate")
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def play(Q):
    env = Adventure()
    s, done, won = env.reset(), False, False
    print("\nGreedy playthrough:")
    while not done:
        a = int(np.argmax(Q[s_idx(s)]))
        print(f"  [{ROOMS[s[0]]}] -> {ACTIONS[a]}")
        s, r, done, won = env.step(a)
    print("  Result:", "WON" if won else "LOST/TIMEOUT")


if __name__ == "__main__":
    EPISODES, ALPHA, GAMMA = 20000, 0.1, 0.95
    Q, rewards, wins = train(EPISODES, ALPHA, GAMMA)
    plot_curves(rewards, wins)

    q_win, q_rew = evaluate(lambda s: int(np.argmax(Q[s_idx(s)])))
    r_win, r_rew = evaluate(lambda s: random.randrange(len(ACTIONS)))

    print(f"Episodes={EPISODES}, alpha={ALPHA}, gamma={GAMMA}")
    print(f"Q-learning agent : win rate {q_win:.1%} | avg reward {q_rew:.1f}")
    print(f"Random baseline  : win rate {r_win:.1%} | avg reward {r_rew:.1f}")
    play(Q)
