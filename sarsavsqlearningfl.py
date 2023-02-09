import numpy as np
import gym
import random
import sys
import atexit
from gym.envs.registration import register


class SARSA:
    def __init__(self):



        '''La fonction __init__ en Python est une méthode de constructeur pour un algorithme d'apprentissage 
        par renforcement dans l'environnement OpenAI Gym. La fonction effectue les opérations suivantes :
        1 Enregistre un nouvel environnement "FrozenLakeNotSlippery-v0" dans OpenAI Gym avec une taille de carte de 4x4 et une propriété de glissement définie à True. L'environnement est configuré pour avoir un nombre maximal de pas par épisode de 100 et un seuil de récompense de 0.8196.
        2 Crée une instance de l'environnement "FrozenLakeNotSlippery-v0".
        3 Initialise la taille de l'espace d'actions et d'états en fonction de l'environnement.
        4 Crée une table Q de taille state_size x action_size avec toutes les valeurs définies à 0.
        5 Définit le nombre total d'épisodes d'entraînement à 20000, le taux d'apprentissage à 0.8, le nombre maximal de pas par épisode à 99 et le facteur de réduction à 0.95.
        6 Définit le taux d'exploration à 1.0 et initialise la probabilité d'exploration à 1.0, la probabilité d'exploration minimale à 0.01 et le taux de décroissance de la probabilité d'exploration à 0.001.
        7 Initialise une liste vide pour stocker les récompenses pour chaque épisode.'''



        register(
            id='FrozenLakeNotSlippery-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name': '4x4', 'is_slippery': True},
            max_episode_steps=100,
            reward_threshold=0.8196,
        )

        self.env = gym.make("FrozenLakeNotSlippery-v0")
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n
        self.qtable = np.zeros((self.state_size, self.action_size))

        self.total_episodes = 20000  # maximal number of training episodes
        self.alpha = 0.8  # learning rate
        self.max_steps = 99  # maximal number of steps per episode
        self.gamma = 0.95  # discount factor

        self.epsilon = 1.0  # exploration rate in epsilon-greedy
        self.max_epsilon = 1.0  # exploration probability at the start
        # minimum exploration probability after exploration probability decay
        self.min_epsilon = 0.01
        self.decay_rate = 0.001  # exploration probability decay rate

        self.rewards = []  # rewards list

    def train(self, total_episodes, alpha, max_steps, gamma, max_epsilon, min_epsilon, decay_rate):
        '''Cette fonction a 7 paramètres :

    total_episodes qui représente le nombre maximal d'épisodes d'entraînement
    alpha qui représente le taux d'apprentissage
    max_steps qui représente le nombre maximal de pas par épisode
    gamma qui représente le facteur de réduction
    max_epsilon qui représente la probabilité d'exploration au début
    min_epsilon qui représente la probabilité minimale d'exploration après la décroissance de la probabilité d'exploration
    decay_rate qui représente le taux de décroissance de la probabilité d'exploration
    Elle utilise l'attribut env de l'objet pour obtenir les informations sur l'espace d'actions et l'espace d'observations et initialise la table de valeur q à zéro.
    La fonction entraîne un algorithme en simulant plusieurs épisodes, en choisissant des actions aléatoirement ou en utilisant la table de valeur q pour déterminer 
    la meilleure action. La probabilité d'exploration est décrémentée à la fin de chaque épisode. Les récompenses totales sont stockées dans une liste et la moyenne
     est affichée en fin de simulation, ainsi que la table de valeur q et la probabilité d'exploration.
    '''




        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n
        self.qtable = np.zeros((self.state_size, self.action_size))
        self.rewards = []
        epsilon = max_epsilon

        for episode in range(total_episodes):
            state = self.env.reset()
            step = 0
            done = False
            total_rewards = 0

            action = self.env.action_space.sample()

            for step in range(max_steps):
                new_state, reward, done, info = self.env.step(action)

                tradeoff = random.uniform(0, 1)

                if tradeoff > epsilon:
                    next_action = np.argmax(self.qtable[new_state, :])
                else:
                    next_action = self.env.action_space.sample()

                self.qtable[state, action] = self.qtable[state, action] + alpha * (
                    reward + gamma * self.qtable[new_state, next_action] - self.qtable[state, action])
                total_rewards = total_rewards + reward
                state = new_state
                action = next_action

                if done == True:
                    break

            epsilon = min_epsilon + \
                (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        self.rewards.append(total_rewards)

        print('score over time:' + str(sum(self.rewards)/total_episodes))
        print(self.qtable)
        print(epsilon)


    def test(self, episodes, max_steps):

        '''Cette fonction de test prend en entrée le nombre d'épisodes et le nombre maximal de pas pour chaque épisode. 
        Pour chaque épisode, la fonction initialise l'état et affiche un message indiquant le numéro de l'épisode. 
        Dans une boucle tant que le nombre de pas est inférieur au nombre maximal de pas, la fonction utilise la méthode render 
        de l'environnement pour afficher l'état actuel et choisit l'action optimale en utilisant la table de valeur q. 
        La fonction effectue l'action choisie, obtient la nouvelle état, la récompense et l'état d'achèvement. 
        Si l'état d'achèvement est "fait", la boucle est interrompue. Enfin, à la fin de chaque épisode, la fonction 
        demande à l'utilisateur s'il souhaite quitter le programme et, s'il répond "O", la boucle est également interrompue.'''



        for episode in range(episodes):
            state = self.env.reset()
            step = 0
            done = False
            print("****************************************************")
            print("EPISODE ", episode)

            while step < max_steps:
                self.env.render()

                action = np.argmax(self.qtable[state, :])

                new_state, reward, done, info = self.env.step(action)

                if done:
                    break
                state = new_state
            choice = input("Voulez-vous quitter le programme? (O/N)")
            if choice.upper() == "O":
                break
            episode += 1


QL = SARSA()
QL.train(20000, 0.8, 99, 0.95, 1.0, 0.01, 0.001)
QL.test(4, 99)
