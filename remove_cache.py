import os
import yaml
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--data_yaml", type=str, help="Data yaml path to remove cache", required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.data_yaml, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    for data_list_txt in data["train"]:
        # extract directory
        data_list_txt = data_list_txt.replace(".txt", ".cache")
        if os.path.exists(data_list_txt):
            # shutil.rmtree(data_list_txt)
            os.remove(data_list_txt)
            print(f"Removed cache: {data_list_txt}")
            
            
if __name__ == "__main__":
    main()