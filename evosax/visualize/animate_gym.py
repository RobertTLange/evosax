from matplotlib import animation
import matplotlib.pyplot as plt
import gym
from ffw_pendulum import ffw_policy, flat_to_network
import numpy as np


def save_frames_as_gif(frames, title, path='./img/', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0,
                        frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.title(title, fontsize=50)
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=30)


if __name__ == '__main__':
    number_gens = [50, 200, 350, 500]
    gen_ckpt = [np.load("experiments/base/gen_50.npy")[0:1, :],
                np.load("experiments/base/gen_200.npy")[0:1, :],
                np.load("experiments/base/gen_350.npy")[0:1, :],
                np.load("experiments/base/gen_500.npy")[0:1, :]]

    for i, weights in enumerate(gen_ckpt):
        shapes = [3, 48, 1]
        jax_weights = flat_to_network(weights, shapes)
        for k, v in jax_weights.items():
            jax_weights[k] = v.squeeze()

        #Make gym env
        env = gym.make('Pendulum-v0')

        #Run the env
        state = env.reset()
        frames = []
        for t in range(2000):
            #Render to frames buffer
            frames.append(env.render(mode="rgb_array"))
            action = np.expand_dims(ffw_policy(jax_weights, state), 1)
            state, reward, done, _ = env.step(action)
            if done:
                break
        env.close()
        save_frames_as_gif(frames, title="Generation " + str(number_gens[i]), filename='gen_' + str(number_gens[i]) + '.gif')
