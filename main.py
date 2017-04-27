from src.Aarohi import Aarohi
# from src import utils

# utils.convertOggToWav("./media")

aarohi = Aarohi()
aarohi.setTrainingData("./media/")
# aarohi.load_model("model1")
# aarohi.train()
aarohi.save_model("model1.h5")
# aarohi.inventSong()