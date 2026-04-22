import yaml, os

class RegisterException(Exception):
    """Raised when attempting to register an invalid or duplicate component."""

class ConfigError(Exception):
    """Raises when config is somwhow malformed"""

"""
INVARIANT?? Reason through this
For any configurable instance, all config-relevant attributes:
- live on the instance
- are mirrored in central_registry
- are updated only via constructor, load, or explicit mutation followed by update()
ALSO IMPORTANT: hide dicts and lists behind a property that retursn by copy? don't want to modify internal structure by accident!
"""
class Config:
    #The constructor for the config class needs to generate the first layer of parameters with the ones coming from a config file
    def __init__(self, filepath: str = None):
        """
        Config class to gather parameters.
        
        :param self: Description
        :param filepath: path to the config.yaml file. If this parameter is not passed, then an empty config object is generated
        :type filepath: str
        """

        self.registered_objects = []

        if filepath is None:
            self.central_registry = {}
            return

        #Create a dict with config params from the filepath
        with open(filepath, "r") as f:
            self.central_registry = yaml.safe_load(f) or {}

            #Make any list into a tuple
            for component_config in self.central_registry.values():
                for k, v in component_config.items():
                    if isinstance(v, list):
                        component_config[k] = tuple(v)
        

    def save(self, dst_filepath: str) -> None:
        with open(dst_filepath, "w") as f:
            yaml.safe_dump(self.central_registry, f, default_flow_style=False)

    #TODO could use a name instead of component_type (that we can deduce from instance nonetheless). This would impreve flexibility
    def register(self, component_type, component_instance, init_params) -> dict:
        """
        Add the parameters of an instance of a configurable class to the registry, in its namespace.\n
        Returns a dictionary holding all parameters relating to the given class *set in the config file*, excluding those overridden by the constructor.
        
        :param component_type: class of the component
        :param component_instance: specific instance of the component
        :param params: kwargs passed to the constructor of the provided instance of the component
        """
        
        if not isinstance(component_instance, component_type):
            raise ValueError("Mismatch between declared component type and type of the component instance provided") 
        for o in self.registered_objects:
            if o.__class__.__name__ == component_type.__name__:
                raise RegisterException("Two objects of the same class cannot be registered")

        if component_instance not in self.registered_objects:
            self.registered_objects.append(component_instance)
        else:
            raise RegisterException("An object cannot be registered more than once")
        
        if component_type.__name__ in self.central_registry.keys():
            obj_registry = self.central_registry[component_type.__name__]
        else:
            obj_registry = {}

        #Check init_params against config_params and return a dict holding config params that were not overridden
        for key, val in init_params.items():
            if key in obj_registry.keys():
                obj_registry.pop(key)
        
        d = obj_registry.copy()
        self.central_registry[component_type.__name__] = obj_registry | init_params #Hold ALL attributes
        
        return d

    """
    @signals RuntimeError when register(obj_type, obj, params) was not called before update
    @ensures central_registry[cls_name] is up to date with the parameters held by the instance
    """
    def update(self, obj, params: dict):
        cls_name = obj.__class__.__name__
        if cls_name not in self.central_registry.keys():
            raise RuntimeError("Tried to update the registry of an object before registering it")

        for name, value in params.items():
            self.central_registry[cls_name][name] = value
        
    #TODO: expand on this. RN it just discards none values
    def is_acceptable(self, value) -> bool:
        """
        return True if the provided value is elegible for being saved in config.
        if the value is not a primitive typer, one with a clear, concise str representation or in general it doesn't make sense to dump it in a .yaml, then return False
        
        :param value: config parameter to filter
        """

        return (value is not None)

