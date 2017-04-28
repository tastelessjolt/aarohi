from src.Aarohi import Aarohi
import numpy as np
# from src import utils

# utils.convertOggToWav("./media")

aarohi = Aarohi()
# aarohi.setTrainingData("./media/")
# aarohi.load_model("model1.h5")
# aarohi.load_model("model2.h5")
aarohi.load_model("model3.h5")
aarohi.train()
aarohi.save_model("model3.h5")
y = aarohi.inventSong()