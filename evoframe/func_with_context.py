class func_with_context:
    def __init__(self, inner_func, context={}):
        self.context = context
        self.inner_func = inner_func

    def __call__(self, *args, **kwargs):
        return self.inner_func(self.context, *args, **kwargs)
