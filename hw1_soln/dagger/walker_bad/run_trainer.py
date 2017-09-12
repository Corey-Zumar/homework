import pickle
import numpy as np
from model import ModelTrainer

if __name__ == "__main__":
    f = open("expert_data.sav", "rb")
    data = pickle.load(f)
    f.close()
    actions = data['actions']
    new_actions = [item[0] for item in actions]
    actions = np.array(new_actions)

    #i = 0
    #new_actions = []
    #while i < len(actions) - 3:
    #    ts = [actions[j][0] for j in range(i, i+3)]
    #    new_actions.append(ts)
    #    i += 3

    #actions = np.array(new_actions, dtype=np.float64)

    #obs = data['observations']
    #new_obs = []
    #i = 0
    #while i < len(obs) - 3:
    #    ts = [obs[j] for j in range(i, i+3)]
    #    new_obs.append(ts)
    #    i += 3
    
    #obs = np.array(new_obs, dtype=np.float64)

    #print(obs.shape)
    obs = data['observations']

#    obs = np.reshape(obs, (obs.shape[0], 1, obs.shape[1]))

    #actions = []
    #for item in data['actions']:
     #   actions.append(item[0])
    #actions = np.array(actions)
    #obs = data['observations']

    trainer = ModelTrainer()
    training_history = trainer.train(obs, actions)
    out = open("training_loss_history.sav", "w")
    pickle.dump(training_history.history["loss"], out)
    out.close()

    out = open("training_accuracy_history.sav", "w")
    pickle.dump(training_history.history["acc"], out)
    out.close()

    trainer.save("trained_model.hd5")
