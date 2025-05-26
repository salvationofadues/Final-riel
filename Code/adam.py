import tensorflow as tf
# Custom Adam optimizer implementation
# This implementation is a simplified version of the Adam optimizer

class CustomAdam(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, name="CustomAdam", **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._slot = {}
        
    def build(self, var_list):
        # Create slots for each variable
        for var in var_list:
            self._slot[id(var)] = {
                "m": self.add_variable_from_reference(var, "m", tf.zeros_like(var)),
                "v": self.add_variable_from_reference(var, "v", tf.zeros_like(var)),
            }


    def update_step(self, grad, var, learning_rate=None):
        if learning_rate is None:
            learning_rate = self._learning_rate
        lr = self._learning_rate
        beta1 = self._beta1
        beta2 = self._beta2
        epsilon = self._epsilon

        slots = self._slot[id(var)]
        m = slots["m"]
        v = slots["v"]

        m.assign(beta1 * m + (1.0 - beta1) * grad)
        v.assign(beta2 * v + (1.0 - beta2) * tf.square(grad))
        
        # Time step

        t = tf.cast(self.iterations + 1, tf.float32)
        m_hat = m / (1.0 - tf.pow(beta1, t))
        v_hat = v / (1.0 - tf.pow(beta2, t))

        var.assign_sub(lr * m_hat / (tf.sqrt(v_hat) + epsilon))

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta1": self._serialize_hyperparameter("beta1"),
            "beta2": self._serialize_hyperparameter("beta2"),
            "epsilon": self.epsilon,
        })
        return config