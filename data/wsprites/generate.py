from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import pickle
import random
random.seed(1543)

from random_walk import RandomWalk
from sprites import sprites_act


def attr_to_str(attr):
    return "_".join(attr.ravel().astype(str).tolist())


def move_frame(im, x, y, pad):
    h, w = im.size
    large = im.resize((h + pad, w + pad), Image.ANTIALIAS)
    mim = large.crop((x, y, x + w, y + h))
    return mim


def loop(seq):
    n = len(seq)
    i = 0
    while True:
        yield seq[i]

        if i == n - 1:
            i = 0
        else:
            i += 1


def generate_actions(n_steps):
    H, W = 96, 128
    h, w = 64, 64

    walk = RandomWalk(H - h, W - w, inertia=0.9, step_size=7)

    walk_actions = []
    global_actions = []
    global_shifts = []
    coords = []
    for _ in range(n_steps):
        shift = walk.step()

        walk_actions.append(walk.d)
        global_actions.append(walk.get_action())
        global_shifts.append(shift)
        coords.append((walk.x, walk.y))

    return walk_actions, global_actions, global_shifts, coords


def generate_video_correlated(sprite_id, actions, coords, attr_str, unique_attrs, X, D, bg=False, max_bg_motion=None):
    H, W = 96, 128

    if bg:
        bg_im = Image.open("./data/wsprites/bg.JPEG")
        w_bg = RandomWalk(max_bg_motion, max_bg_motion, inertia=0.95, step_size=2)

    sprite_indices = attr_str == unique_attrs[sprite_id]
    sprite_actions = D[sprite_indices]
    sprite_videos = X[sprite_indices]

    sprite_action_to_video = {
        a: sprite_videos[np.where(sprite_actions.argmax(axis=-1)[:, 0] == a)[0].item()] for a in range(9)
    }

    action_generator = {
        a: loop(sprite_action_to_video[a]) for a in range(9)
    }

    available_actions = {
        "r": [2],
        "l": [1],
        "u": [0],
        "d": [0],
        "s": [3, 4, 5, 6, 7, 8]
    }

    walk_video = []
    masks = []
    local_actions = []
    bg_shifts = []
    i = 0
    prev_a = None
    while i < len(actions):
        cur_action = actions[i]
        c = coords[i]

        a = random.choice(available_actions[cur_action])
        if cur_action == "s":
            while a == prev_a:
                a = random.choice(available_actions[cur_action])
        prev_a = a

        if cur_action == "s":
            max_steps = random.choice([4, 5, 6, 7, 8])
            j = 0
            while i + j < len(actions) and actions[i + j] == "s" and j < max_steps:
                cur_action = actions[i + j]
                c = coords[i + j]

                if j == 0:
                    action_generator[a] = loop(sprite_action_to_video[a])
                frame = next(action_generator[a])
                frame = (255 * frame).astype(np.uint8)

                im = Image.new("RGB", (W, H))
                frame = Image.fromarray(frame)

                im.paste(frame, (c[0], c[1]))
                mask = (np.array(im) > 0).astype(np.float32)

                if bg:
                    im = np.array(im)

                    bg_shift = w_bg.step()
                    bg_shifts.append(bg_shift)
                    x = w_bg.x
                    y = w_bg.y
                    m_bg = move_frame(bg_im.copy(), x, y, max_bg_motion)
                    m_bg = m_bg.resize((W, H), Image.ANTIALIAS)

                    bg_c = np.array(m_bg.copy())

                    comb = mask * im + (1 - mask) * bg_c
                    comb_im = Image.fromarray(comb.astype(np.uint8))

                    im = comb_im.copy()

                local_actions.append(a)
                walk_video.append(im)
                masks.append(Image.fromarray((255 * mask).astype(np.uint8)).convert("L"))

                j += 1
            i = i + j - 1
        else:
            frame = next(action_generator[a])
            frame = (255 * frame).astype(np.uint8)

            im = Image.new("RGB", (W, H))
            frame = Image.fromarray(frame)

            im.paste(frame, (c[0], c[1]))
            mask = (np.array(im) > 0).astype(np.float32)

            if bg:
                im = np.array(im)
                bg_shift = w_bg.step()
                bg_shifts.append(bg_shift)
                x = w_bg.x
                y = w_bg.y
                m_bg = move_frame(bg_im.copy(), x, y, max_bg_motion)
                m_bg = m_bg.resize((W, H), Image.ANTIALIAS)

                bg_c = np.array(m_bg.copy())

                comb = mask * im + (1 - mask) * bg_c
                comb_im = Image.fromarray(comb.astype(np.uint8))

                im = comb_im.copy()

            local_actions.append(a)
            walk_video.append(im)
            masks.append(Image.fromarray((255 * mask).astype(np.uint8)).convert("L"))

        i += 1

    if bg:
        return walk_video, masks, local_actions, bg_shifts
    else:
        return walk_video, masks, local_actions


def main():
    X_train, X_test, A_train, A_test, D_train, D_test = sprites_act('./data/wsprites/sprites/', return_labels=True)

    X = np.concatenate([X_train, X_test], axis=0)
    A = np.concatenate([A_train, A_test], axis=0)
    D = np.concatenate([D_train, D_test], axis=0)

    attr_str = []
    for a in A:
        attr_str.append(attr_to_str(a))
    attr_str = np.array(attr_str)

    unique_attrs = np.unique(attr_str)

    os.mkdir("./data/wsprites/data/train")
    os.mkdir("./data/wsprites/data/val")
    os.mkdir("./data/wsprites/data/test")

    videos_per_sprite = 10
    train_size = 0.8
    val_size = 0.1
    assert train_size + val_size < 1, "Dataset splits sizes must sum up to 1"
    for sprite_id in tqdm(range(unique_attrs.shape[0])):
        for i in range(videos_per_sprite):
            if sprite_id < train_size * unique_attrs.shape[0]:
                data_path = "./data/wsprites/data/train"
            elif sprite_id < (train_size + val_size) * unique_attrs.shape[0]:
                data_path = "./data/wsprites/data/val"
            else:
                data_path = "./data/wsprites/data/test"
            folder = os.path.join(data_path, "sprite_{:06}_video_{:03}".format(sprite_id, i))
            os.mkdir(folder)
            os.mkdir(os.path.join(folder, "masks"))
            walk_actions, global_actions, global_shifts, coords = generate_actions(n_steps=90)
            walk_video, masks, local_actions, bg_shifts = generate_video_correlated(
                sprite_id, walk_actions, coords, attr_str, unique_attrs, X, D, bg=True, max_bg_motion=25)
            actions = {
                "global": global_actions,
                "local": local_actions,
                "shifts": global_shifts,
                "bg_shifts": bg_shifts,
            }
            with open(os.path.join(folder, "glactions.pkl"), "w") as f:
                pickle.dump(actions, f)
            with open(os.path.join(folder, "actions.pkl"), "w") as f:
                pickle.dump(global_actions, f)
            with open(os.path.join(folder, "dones.pkl"), "w") as f:
                pickle.dump([None] * len(global_actions), f)
            with open(os.path.join(folder, "rewards.pkl"), "w") as f:
                pickle.dump([None] * len(global_actions), f)
            with open(os.path.join(folder, "metadata.pkl"), "w") as f:
                pickle.dump([None] * len(global_actions), f)
            for j, frame in enumerate(walk_video):
                frame.save(os.path.join(folder, "{:05}.png".format(j)))
                masks[j].save(os.path.join(folder, "masks/{:05}.png".format(j)))


if __name__ == "__main__":
    main()