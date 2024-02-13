import yaml

class Config:
    def __init__(self, path):
        self.CFG = self._loadConfig(path)

    def _loadConfig(self, path):
        with open(path, 'r', encoding='utf-8') as yamlfile:
            cfg = yaml.safe_load(yamlfile)
        return cfg

    def get(self, *keys):
        value = self.CFG
        for key in keys:
            try:
                value = value[key]
            except (KeyError, TypeError):
                return None
        return value
     
    def __str__(self):
        # Return the YAML formatted string for prettier printing
        return yaml.dump(self.CFG, default_flow_style=False, sort_keys=True, indent=4, allow_unicode=True)

    def __repr__(self):
        return self.__str__()