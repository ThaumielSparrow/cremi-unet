from preprocess import CREMI
from train import main as train_model


def main():
    container = CREMI(samplefolder='samples/train/', savefolder='data/train/', autocon=True)
    container.preprocess()
    container.test_train_split(train_folder='data/train/', test_folder='data/test/', train_volume=0.8)
    train_model()


if __name__ == '__main__':
    main()
