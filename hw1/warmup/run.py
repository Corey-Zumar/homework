import pickle
import numpy as np
from model import ModelTrainer

if __name__ == "__main__":
    f = open("expert_data.sav", "rb")
    data = pickle.load(f)
    f.close()
    actions = []
    for item in data['actions']:
        actions.append(item[0])
    actions = np.array(actions)
    obs = []
    for item in data['observations']:
        obs.append([[item[i]] for i in range(len(item))])
    obs = np.array(obs)

    trainer = ModelTrainer()
    training_history = trainer.train(obs, actions)


