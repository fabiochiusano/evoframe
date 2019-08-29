class AgentWrapper:
    def __init__(self, model):
        self.model = model
        self.predict_func = None

    def predict(self, inputs):
        return self.predict_func(self.model, inputs)

def get_agent_wrapper_func(predict_func):
    predict_func = predict_func
    def get_agent_wrapper_aux(model):
        agent_wrapper = AgentWrapper(model)
        agent_wrapper.predict_func = predict_func
        return agent_wrapper
    return get_agent_wrapper_aux
