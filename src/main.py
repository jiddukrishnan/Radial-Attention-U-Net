from src.train import main
import yaml

if __name__ == '__main__':
    with open('configs/default.yaml') as fp:
        cfg = yaml.safe_load(fp)
    main(cfg)
