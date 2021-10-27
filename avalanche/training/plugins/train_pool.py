from skmultiflow.drift_detection import ADWIN

from avalanche.training.plugins.strategy_plugin import StrategyPlugin


class TrainPoolPlugin(StrategyPlugin):
    def __init__(self):
        self.net = None
        # self.loss = None
        # self.optimizer = None
        super().__init__()
        print(self)

    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.call_predict = False
        strategy.model.mb_yy = strategy.mb_y

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.mb_yy = None

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    @staticmethod
    def add_to_frozen_pool(strategy: 'BaseStrategy'):
        strategy.model.add_nn_with_lowest_loss_to_frozen_list()
        strategy.model.reset()

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        if strategy.model.auto_detect_tasks:
            pass
        else:
            self.add_to_frozen_pool(strategy)
        strategy.model.training_exp += 1

    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.print_stats(after_training=True)

    def before_eval(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.load_frozen_pool()

    def before_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.call_predict = True
        strategy.model.mb_yy = strategy.mb_y
        strategy.model.mb_task_id = strategy.mb_task_id

    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.mb_yy = None

    def after_eval(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.print_stats(after_training=False)
        strategy.model.clear_frozen_pool()
