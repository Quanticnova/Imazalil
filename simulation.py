
import yaml

with open("simconfig.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

print(cfg['Sim']['Pmate']['Pred'])
