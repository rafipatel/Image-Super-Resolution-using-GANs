from utils import create_data_lists

if __name__ == "__main__":
    create_data_lists(train_folders = ["../train2014"], test_folders = ["../Set5_"], min_size = 100, output_folder = "./")

    # choose model from config.yaml..., run relevant train.py