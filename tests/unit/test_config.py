from src.spotmicro.tools.config import Config, RegisterException

import unittest, tempfile, yaml, os

"""
This file should test Config as a standalone class, without considering its interactions with the "configurable" decorator. Therefore, parameters for the Dummy class are to be treated as dictionaries, since Config works with them
"""

class Dummy:
    pass

class DumDum:
    pass

class TestConfigBasic(unittest.TestCase):
    #<----- BASIC FUNCTIONALITIES ----->
    def test_empty_construction(self):
        config = Config()
        self.assertEqual(config.central_registry, {})
        self.assertEqual(config.registered_objects, [])
    
    def test_construction_from_file(self):
        data = {
            "Dummy": {
                "a": 1,
                "b": 3
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            yaml.safe_dump(data, f)
            path = f.name

        try:
            cfg = Config(path)
            self.assertEqual(cfg.central_registry, data)
        finally:
            os.remove(path)
    
    def test_register_new_component(self):
        cfg = Config()
        obj1 = Dummy()
        obj2 = DumDum()

        params = {"a": 1, "b": 2}
        returned = cfg.register(Dummy, obj1, params)

        self.assertEqual(returned, {})
        self.assertIn(obj1, cfg.registered_objects)
        self.assertEqual(len(cfg.registered_objects), 1)
        self.assertEqual(cfg.central_registry["Dummy"], params)

        params2 = {"c" : "yeye"}
        expected_registry = {
            "Dummy" : {
                "a" : 1,
                "b" : 2
            },
            "DumDum" : {
                "c" : "yeye"
            }
        }
        ret2 = cfg.register(DumDum, obj2, params2)
        self.assertEqual(ret2, {})
        self.assertIn(obj2, cfg.registered_objects)
        self.assertIn(obj1, cfg.registered_objects) #What was in before is still in
        self.assertEqual(len(cfg.registered_objects), 2)
        self.assertDictEqual(cfg.central_registry, expected_registry)
    

    def test_register_overrides_file_params(self):
        initial = {
            "Dummy": {
                "a": 1,
                "b": 2
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            yaml.safe_dump(initial, f)
            path = f.name

        try:
            cfg = Config(path)
            obj = Dummy()

            overrides = {"b": 99}
            remaining = cfg.register(Dummy, obj, overrides)

            self.assertEqual(remaining, {"a": 1})
            self.assertEqual(cfg.central_registry["Dummy"], {"a": 1, "b": 99})
        finally:
            os.remove(path)
    

    def test_update(self):
        cfg = Config()
        obj = Dummy()

        cfg.register(Dummy, obj, {"a": 1})
        cfg.update(obj, {"a": 2, "c": 3})

        self.assertEqual(
            cfg.central_registry["Dummy"],
            {"a": 2, "c": 3}
        )

    def test_save(self):
        cfg = Config()
        obj = Dummy()
        cfg.register(Dummy, obj, {"x": 10})

        with tempfile.NamedTemporaryFile(mode="r", delete=False) as f:
            path = f.name

        try:
            cfg.save(path)

            with open(path, "r") as f:
                data = yaml.safe_load(f)

            self.assertEqual(data, {"Dummy": {"x": 10}})
        finally:
            os.remove(path)

    #<----- EDGE CASES ----->
    def test_empty_yaml_file(self):
        # Create an empty file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            path = f.name

        try:
            cfg = Config(path)
            # Should create an empty central_registry
            self.assertEqual(cfg.central_registry, {})
            self.assertEqual(cfg.registered_objects, [])
        finally:
            os.remove(path)

    def test_invalid_inputs(self):
        #Instantiation with non existent file?
        with self.assertRaises(FileNotFoundError):
            Config("nonExistent.yaml")
    
    def test_double_object(self):
        cfg = Config()
        o1 = Dummy()
        o2 = Dummy()

        p1 = {"a": 1, "b": 2}
        p2 = {"a": 3, "b": 4}

        r1 = cfg.register(Dummy, o1, p1)
        with self.assertRaises(RegisterException):
            r2 = cfg.register(Dummy, o2, p2)
    
    def test_empty_params(self):
        cfg = Config()
        o = Dummy()
        p = {}
        expected_registry = {
            "Dummy": {}
        }

        r = cfg.register(Dummy, o, p)
        self.assertDictEqual(r, {})
        self.assertDictEqual(cfg.central_registry, expected_registry)

    def test_invalid_usage(self):
        cfg = Config()
        o = Dummy()
        p1 = {"a": 1}

        with self.assertRaises(RuntimeError):
            cfg.update(o, p1)
        #Test that state of config was not modified
        self.assertDictEqual(cfg.central_registry, {})
        self.assertEqual(len(cfg.registered_objects), 0)
    
    def test_save_empty(self):
        cfg = Config()
        o = Dummy()
        p1 = {}

        with tempfile.NamedTemporaryFile(mode="r", delete=False) as f:
            path = f.name

        try:
            cfg.save(path)

            with open(path, "r") as f:
                data = yaml.safe_load(f)

            self.assertEqual(data, {})
        finally:
            os.remove(path)
    