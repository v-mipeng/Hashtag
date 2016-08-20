from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping, Momentum, Scale

class WAdaDelta(AdaDelta):
    def __init__(self, decay_rate=0.95, epsilon=1e-6, special_para_names = None):
        self.special_para_names = special_para_names
        super(WAdaDelta, self).__init__(decay_rate=0.95, epsilon=1e-6)

    def compute_step(self, parameter, previous_step):
        if parameter.name in self.special_para_names:
            step = 0.1*previous_step
            update = []
            return step, update
        else:
            return super(WAdaDelta, self).compute_step(parameter, previous_step)