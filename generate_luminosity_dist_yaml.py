"""
Helper script to generate luminosity distances for ANTs and save to luminosity_dist.yaml
"""
import os
import yaml
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


# Load ANT_proper_redshift_dict from metadata/redshift.yaml
redshift_yaml_path = os.path.join("metadata", "redshifts.yaml")
with open(redshift_yaml_path, "r") as f:
    ANT_proper_redshift_dict = yaml.safe_load(f)

# Cosmology assumptions
H0 = 70  # km/s/Mpc
om_M = 0.3
fcdm = FlatLambdaCDM(H0=H0, Om0=om_M)

luminosity_dist_dict = {}
for ant, z in ANT_proper_redshift_dict.items():
    d = fcdm.luminosity_distance(z).to(u.cm).value
    luminosity_dist_dict[ant] = float(d)

# Save to YAML
config_dir = "config"
os.makedirs(config_dir, exist_ok=True)
lum_yaml_path = os.path.join("metadata", "luminosity_dist.yaml")
with open(lum_yaml_path, "w") as f:
    yaml.dump(luminosity_dist_dict, f)

print(f"Saved luminosity distances to {lum_yaml_path}")
