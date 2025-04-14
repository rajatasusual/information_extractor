from importlib import resources

with resources.files("information_extractor.assets").joinpath("config.json").open("r") as f:
    config = json.load(f)
