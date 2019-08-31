import inspect
from abc import ABC, abstractmethod

MODEL_OPERATOR_PREFIX = "es_"

class Model(ABC):
    @abstractmethod
    def predict(self, inp):
        pass

    def get_operators(self):
        model_operators = [m for m in dir(self) if m.startswith(MODEL_OPERATOR_PREFIX)]
        model_operators_summary = []
        for operator in model_operators:
            operator_method = getattr(self, operator)
            operator_arg_spec = inspect.getfullargspec(operator_method)
            operator_args = operator_arg_spec.args[1:] # do not consider 'self' as argument
            operator_defaults = operator_arg_spec.defaults
            model_operator_summary = operator + " "
            for i,arg in enumerate(operator_args):
                if i < len(operator_args) - len(operator_defaults):
                    model_operator_summary += arg + " "
                else:
                    model_operator_summary += arg + "=" + str(operator_defaults[i - (len(operator_args) - len(operator_defaults))]) + " "
            model_operator_summary = model_operator_summary.rstrip()
            model_operators_summary.append(model_operator_summary)
        return model_operators_summary
