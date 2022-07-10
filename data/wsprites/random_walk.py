import random

random.seed(1543)


class RandomWalk:
    def __init__(self, h, w, inertia=0.8, step_size=3):
        self.h = h
        self.w = w
        self.inertia = inertia
        self.step_size = step_size

        self.x = None
        self.y = None
        self.d = None

        self.mapping = {v: i for i, v in enumerate(["l", "r", "u", "d", "s"])}

        self.reinit_state()

    def reinit_state(self):
        self.x = random.randint(0, self.w - 1)
        self.y = random.randint(0, self.h - 1)

    def actions(self, exclude_cur=False):
        actions = {"l": True, "r": True, "u": True, "d": True, "s": True}

        if self.x == self.w - 1:
            actions["r"] = False
        if self.x == 0:
            actions["l"] = False
        if self.y == self.h - 1:
            actions["d"] = False
        if self.y == 0:
            actions["u"] = False

        if exclude_cur:
            actions[self.d] = False

        return [k for k, v in actions.items() if v]

    def sample_action(self):
        cur_actions = self.actions(exclude_cur=False)
        new_d = random.choice(cur_actions)

        if self.d is None:
            self.d = new_d
            return
        if self.d not in cur_actions:
            self.d = new_d
            return
        if len(cur_actions) == 1:
            return

        cur_actions = self.actions(exclude_cur=True)
        new_d = random.choice(cur_actions)

        p = random.uniform(0, 1)

        if self.d != "s" and p > self.inertia:
            self.d = new_d
        elif self.d == "s" and p > 0.9:
            self.d = new_d

    def get_action(self):
        return self.mapping[self.d]

    def step(self):
        self.sample_action()

        if self.d == "l":
            prev_x = self.x
            self.x = max(0, self.x - self.step_size)
            return 2.0 * (self.x - prev_x) / self.w, 0

        if self.d == "r":
            prev_x = self.x
            self.x = min(self.w - 1, self.x + self.step_size)
            return 2.0 * (self.x - prev_x) / self.w, 0

        if self.d == "u":
            prev_y = self.y
            self.y = max(0, self.y - self.step_size)
            return 0, 2.0 * (self.y - prev_y) / self.h

        if self.d == "d":
            prev_y = self.y
            self.y = min(self.h - 1, self.y + self.step_size)
            return 0, 2.0 * (self.y - prev_y) / self.h

        if self.d == "s":
            return 0, 0
