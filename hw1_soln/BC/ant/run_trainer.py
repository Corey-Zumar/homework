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
    obs = data['observations']

    trainer = ModelTrainer()
    training_history = trainer.train(obs, actions)
    out = open("training_loss_history.sav", "w")
    pickle.dump(training_history.history["loss"], out)
    out.close()

    out = open("training_accuracy_history.sav", "w")
    pickle.dump(training_history.history["acc"], out)
    out.close()

    trainer.save("trained_model.hd5")
