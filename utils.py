import yaml
from pyriemann.tangentspace import TangentSpace
from CustomToTangentSpace import CustomToTangentSpace


def read_yaml(yaml_path):
    """Open and read safely a yaml file."""
    with open(yaml_path, 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)
        except :
            print("Couldn't load yaml file: {}.".format(yaml_path))
            quit()
    return parameters


def load_metrics():
    content = read_yaml('metrics.yml')
    print(content)

    pyr_metrics = [TangentSpace(metric=m) for m in content['pyriemann']]
    geo_metrics = [CustomToTangentSpace(geometry=m) for m in content['geomstats']]

    return pyr_metrics + geo_metrics


if __name__ == '__main__':
    r = load_metrics()
    print(r)
