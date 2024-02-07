class BaseVisualizer:
    def __init__(self, config):
        self.config = config
        self._init_visualizer()

    def _init_visualizer(self):
        raise NotImplementedError
