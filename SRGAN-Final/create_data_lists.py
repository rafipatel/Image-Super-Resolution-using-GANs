from utils import create_data_lists

if __name__ == "__main__":
    create_data_lists(train_folders = ["../test2015"], val_folders = ["../val2017"], test_folders = ["../Set5", "../Set14", "../BSDS100"], min_size = 100, output_folder = "./")