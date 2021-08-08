from skmultiflow.drift_detection import ADWIN

from avalanche.training.plugins.strategy_plugin import StrategyPlugin


class TrainPoolPlugin(StrategyPlugin):
    def __init__(self):
        self.net = None
        # self.loss = None
        # self.optimizer = None
        self.loss_estimator = ADWIN(delta=1e-5)
        super().__init__()

    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.call_predict = False
        strategy.model.mb_yy = strategy.mb_y

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.mb_yy = None

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        pre_update_loss_estimation = self.loss_estimator.estimation
        self.loss_estimator.add_element(strategy.loss.item())
        if self.loss_estimator.detected_change() and self.loss_estimator.estimation > pre_update_loss_estimation:
            strategy.model.samples_seen_for_train_after_drift = 0
            print('Change detected. Training ALL NNs')

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.add_nn_with_lowest_loss_to_frozen_list()
        strategy.model.reset()

    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.print_stats(after_eval=False)

    def before_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.call_predict = True
        strategy.model.mb_yy = strategy.mb_y
        strategy.model.mb_task_id = strategy.mb_task_id

    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.mb_yy = None

    def after_eval(self, strategy: 'BaseStrategy', **kwargs):
        strategy.model.print_stats(after_eval=True)
