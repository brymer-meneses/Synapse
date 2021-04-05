
# Contains a class that can decorate a function 
# and give it a name. This makes it easier 
# to understand the gradient graph of a backward 
# pass

# https://stackoverflow.com/questions/22797580/how-to-replace-str-for-a-function
class __SetString:

    def __init__(self, function, name: str = None) -> None: #type: ignore
        self.name = name
        self.function = function

    def __call__(self, *args, **kwargs): #type: ignore
        return self.function(*args, **kwargs)

    def __str__(self) -> str:
        return self.name

class Named:
    def __init__(self,name=None):
        if name is not None:
           self.name = name
    def __call__(self,function):
        if self.name is None:
            self.name = function.__name__

        return __SetString(function, self.name)


