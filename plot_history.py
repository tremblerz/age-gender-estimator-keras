import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="Plots training curves from history file.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to history h5 file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_path = args.input

    df = pd.read_hdf(input_path, "history")
    input_dir = os.path.dirname(input_path)
    plt.plot(df["gender_loss"], label="loss (gender)")
    plt.plot(df["age_loss"], label="loss (age)")
    plt.plot(df["val_gender_loss"], label="val_loss (gender)")
    plt.plot(df["val_age_loss"], label="val_loss (age)")
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(input_dir, "loss2.png"))
    plt.cla()

    plt.plot(df["gender_acc"], label="accuracy (gender)")
    plt.plot(df["age_acc"], label="accuracy (age)")
    plt.plot(df["val_gender_acc"], label="val_accuracy (gender)")
    plt.plot(df["val_age_acc"], label="val_accuracy (age)")
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(os.path.join(input_dir, "accuracy2.png"))


if __name__ == '__main__':
    main()
