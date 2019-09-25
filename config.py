import yaml

with open("config.yml",'r') as ymlfile:
	cfg = yaml.load(ymlfile)

for section in cfg:
	print(section)
print cfg['Num_layers']
print cfg['Layers']
print cfg['LossFunc']
print cfg['Optimizer']
