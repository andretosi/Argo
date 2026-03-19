from .config import Config, ConfigError
import inspect, yaml, os
from functools import wraps


def configurable(cls):
    """
    Decorator function that extends the behaviour of the decorated class by wrapping its init and adding a config attribute.
    It injects the logic to extract kwargs from the class constructor and their defaults in the init, to then pass them to the central registry to hold.
    This class treats all kwargs parameters as config parameters, besides those whose default value is None or which are not trivially serializable.

    WARNING: it is not currently possible to subscribe 2 configurable object of the same class to the same config object. 
    
    :param cls: This class is any object that can be configured, and as such should be linked to the central registry
    """
    
    original_init = cls.__init__
    init_sig = inspect.signature(original_init) #extract consstructor signature so that we can copypaste it on our wrapper

    sig_params = init_sig.parameters

    # 1. Ensure config file exists
    if "config" not in sig_params:
        raise ValueError(
            f"Every configurable class must have a parameter named 'config' ({cls})"
        )

    # 2. Ensure `config` is required (no default)
    if sig_params["config"].default is not inspect.Parameter.empty:
        raise ValueError(
            f"Parameter 'config' must be required (no default) in ({cls})"
        )

    # 3. Collect configurable (defaulted) parameters, skipping self and None
    excluded_params = set(getattr(cls, "__config_exclude__", ()))

    default_params = {
        name: param.default
        for name, param in sig_params.items()
        if name not in ("self", "config")
        and name not in excluded_params
        and param.default is not (inspect.Parameter.empty or None)
    }
    

    #<----- WRAPPER ----->
    @wraps(original_init)#used to preserve everything about the wrapped class and pass it to the wrapper
    def __init__(self, *args, **kwargs):
        """
        This is a wrapper constructor that basically injects some config logic around the constructor of the given class
        """
        #These lines do not modify the class!!
        bound = init_sig.bind(self, *args, **kwargs) #Map positional and keyword arguments to the corect parameter. Returns a dict
        bound.apply_defaults()#Fill the above map with default values when none were provided
        self.config = bound.arguments["config"]

        if not isinstance(bound.arguments["config"], Config):
            raise TypeError(f"config parameter must be of Config type, {type(self.config)}was given")
        
        #extract params and their runtime value
        overridden_params = {
            name : value for name, value in kwargs.items()
            if name in default_params
            and name not in excluded_params
            and self.config.is_acceptable(value)
        }
        
        original_init(self, *args, **kwargs) #run the original constructor

        config_parameters = self.config.register(cls, self, overridden_params) #register the initialized instance to config
        for c_param in config_parameters.keys():
            if c_param not in default_params.keys():
                raise ConfigError(f"Parameter {c_param} found in config file provided for {self.__class__.__name__} is invalid, because it was not defined in the constructor of the aforementioned object")
        
        #bind all config parameters to attributes
        for name, value in config_parameters.items():
            setattr(self, name, value)
        
    #<----- END WRAPPER ----->

    #<----- SAVE LOGIC ------>
    def save(self, path : str):
        """
        Save the parameters of **this** configurable objects to the given file

        :param path: path to the file to dump the config in.
        :type path: str
        """
        if not os.path.exists(path):
            open(path, "w").close()  # create empty file
            cfg = {}
        else:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f) or {}

        cls_name = self.__class__.__name__
        if cls_name not in cfg.keys():
            cfg[cls_name] = {}
 
        cfg[cls_name] = cfg[cls_name] | self.config.central_registry[cls_name]
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
    #<----- END SAVE   ------>
    
    #<----- LOAD LOGIC ------>
    #TODO: should we add the option to not override params defined at construction? This might be an entirely different method, like merge
    def load(self, path: str):
        """
        Override the current config for this component with the parameters defined in another config file.\n
        Note that loading a new config will override any parameter already set explicitly, if present in the new config

        :param path: Description
        :type path: str
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File \"{path}\" not found")
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        
        cls_name = self.__class__.__name__
        if cls_name not in cfg.keys():
            raise ValueError(f"Given config file ({path}) has no appropriate section for this object ({cls_name})")

        for name, value in cfg[cls_name].items():
            setattr(self, name, value)
        
        self.config.update(self, cfg[cls_name])
    #<----- END LOAD   ------>


    #change the class constructor with our wrapper and add functions
    cls.__init__ = __init__
    cls.load = load
    cls.save = save

    return cls
