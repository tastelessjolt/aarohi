from src.Aarohi import Aarohi
from src import utils

# utils.convertOggToWav("./media")

aarohi = Aarohi()
aarohi.setTrainingData("./media/")
aarohi.train()
aarohi.inventSong()