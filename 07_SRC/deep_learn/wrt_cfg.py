import yaml
from pathlib import Path

# Default FEUNet configuration in YAML format
feunet_config = {
    "in_channels": 1,
    "out_channels": 4,
    "feature_channels": [16, 32, 64, 128, 256],
    "dropouts": [0.0, 0.0, 0.0, 0.0, 0.0],
    "use_glca": True,
    "use_fusion": True,
    "bilinear": True,
    "glca_levels": [1, 2, 3,],
}

config_path = Path("feunet_config.yaml")
with config_path.open("w") as f:
    yaml.dump(feunet_config, f)

config_path
