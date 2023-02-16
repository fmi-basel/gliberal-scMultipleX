import argparse
import configparser


def conf_to_dict(config):
    return {"p": config["DEFAULT"]["p"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    kwargs = conf_to_dict(config)

    print(kwargs)


if __name__ == "__main__":
    # python ./scripts/extract_well_features.py --config
    # scripts/configs/extract_well_features.conf
    main()
