"""
This module defines two classes: Parameterization and Transformation.
A parameterization is a simple and consistent way to generate a random parameterization for a given transformation.
The transformation abstract class defines the interface for all transformations.
    A transformation must have a
    1. Unique identifier string that is the same for all instances of the same transformation
    2. Parameterization size that is the same for all instances of the same transformation
    3. An abstract transform method that takes an input and parameterization and returns a transformed input
    3. __call__ method that takes input data and calls the transform method with the input and the parameterization
    It has a constructor that generates a parameterization based on a seed.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple

class RandomGenerator(ABC):
    """
    A superclass for parameter generators
    Always returns the
    """
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize a new instance of the random generator class
        :param seed: The seed to use for the random number generator
        """
        self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        """
        Seeds the random number generator
        :param seed: The seed to use for the random number generator
        """
        if seed is not None:
            np.random.seed(seed)
        self.value = self.generate()

    @abstractmethod
    def generate(self) -> Any:
        """
        An abstract method that generates a random parameter
        :return: The generated parameter
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def extremes(self) -> List[Any]:
        """
        An abstract method that returns three values: the minimum value, the maximum value, and an average example value
        :return: The minimum value, an average example value, and the maximum value
        """
        raise NotImplementedError
    
    def __call__(self):
        """
        Call method that generates a random parameter
        :return: The generated parameter
        """
        return self.value
    

class Parameterization:
    """
    A simple class that provides a consistent way to generate the parameters of a transformation.
    Produces a dictionary of parameter values for a given transformation given a list of parameter ids.
    """
    def __init__(self, parameter_ids: List[str], param_generators: List[RandomGenerator], seed: Optional[int] = None, override: Optional[int] = None):
        self.size = len(parameter_ids)
        self.parameter_ids = parameter_ids
        self.param_generators = param_generators
        self.generate_parameterization(seed, override)

    def generate_parameterization(self, seed: Optional[int] = None, override: Optional[int] = None):
        """
        Generates a parameterization for a transformation
        :param seed: The seed to use for the random number generator
        :param override: Sets all values of the parameterization to the given value
        :return: The parameterization
        """
        seed_generator = np.random.default_rng(seed)
        if override is not None:
            # parameterization_values = np.full(self.size, override)
            assert type(override) == int and override >= 0 and override < 3
            parameterization_values = [generator.extremes[override] for generator in self.param_generators]
            self.parameterization = dict(zip(self.parameter_ids, parameterization_values))
            return
        for generator in self.param_generators:
            gen_seed = seed_generator.integers(0, 2**32 - 1)
            generator.seed(gen_seed)
        parameterization_values = [generator() for generator in self.param_generators]
        self.parameterization = dict(zip(self.parameter_ids, parameterization_values))

    def get_parameterization(self) -> Dict[str, float]:
        """
        :return: The parameterization
        """
        return self.parameterization
    
    def __getitem__(self, key):
        """
        :param key: The key to use to access the parameterization
        :return: The value of the parameterization at the given key
        """
        return self.parameterization[key]
    
    def __repr__(self):
        """
        :return: A string representation of the parameterization
        """
        return str(self.parameterization)


class DefaultGenerator(RandomGenerator):
    """
    Returns a number in the range [0, 1)
    """
    def generate(self) -> Any:
        """
        Generates a random parameter
        :return: The generated parameter
        """
        return np.random.rand()
    
    @property
    def extremes(self) -> List[Any]:
        """
        Returns the minimum value, the maximum value, and an average example value
        :return: The minimum value, the maximum value, and an average example value
        """
        return [0, 0.5, 0.9999]

class ApplicationGenerator(RandomGenerator):
    """
    Applies a function passed in to a random parameter [0, 1)
    Used when there is no appropriate default generator
    """
    def __init__(self, func, seed: Optional[int] = None):
        """
        Applies a function passed in to a random parameter [0, 1)
        Used when there is no appropriate default generator
        :param func: The function to apply to the random parameter
        :param seed: The seed to use for the random number generator
        """
        self.func = func
        super().__init__(seed)
    
    def generate(self) -> Any:
        """
        Generates a random parameter
        :return: The generated parameter
        """
        raw = np.random.rand()
        val = self.func(raw)
        return val
    
    @property
    def extremes(self) -> List[Any]:
        """
        Returns the minimum value, the maximum value, and an average example value
        :return: The minimum value, the maximum value, and an average example value
        """
        return self.func(0), self.func(0.5), self.func(0.999)

class IntRangeGenerator(RandomGenerator):
    """
    Returns a number in the range [min_val, max_val)
    """
    def __init__(self, min_val: int, max_val: int, seed: Optional[int] = None):
        """
        Initialize a new instance of the random generator class
        :param min_val: The minimum value of the range
        :param max_val: The maximum value of the range
        :param seed: The seed to use for the random number generator
        """
        assert min_val < max_val
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(seed)
    
    def generate(self) -> Any:
        """
        Generates a random parameter
        :return: The generated parameter
        """
        return np.random.randint(self.min_val, self.max_val)
    
    @property
    def extremes(self) -> List[Any]:
        """
        Returns the minimum value, the maximum value, and an average example value
        :return: The minimum value, the maximum value, and an average example value
        """
        return [self.min_val, (self.min_val + self.max_val) // 2, self.max_val]
    
class FloatRangeGenerator(RandomGenerator):
    """
    Returns a number in the range [min_val, max_val)
    """
    def __init__(self, min_val: float, max_val: float, seed: Optional[int] = None):
        """
        Initialize a new instance of the random generator class
        :param min_val: The minimum value of the range
        :param max_val: The maximum value of the range
        :param seed: The seed to use for the random number generator
        """
        assert min_val < max_val
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(seed)
    
    def generate(self) -> Any:
        """
        Generates a random parameter
        :return: The generated parameter
        """
        return np.random.uniform(self.min_val, self.max_val)
    
    @property
    def extremes(self) -> List[Any]:
        """
        Returns the minimum value, the maximum value, and an average example value
        :return: The minimum value, the maximum value, and an average example value
        """
        return [self.min_val, (self.min_val + self.max_val) / 2, self.max_val]

class ChoiceGenerator(RandomGenerator):
    """
    Returns a random choice from a list of choices
    """
    def __init__(self, choices: List[Any], seed: Optional[int] = None):
        """
        Initialize a new instance of the random generator class
        :param choices: The list of choices to choose from
        :param seed: The seed to use for the random number generator
        """
        assert len(choices) > 0
        self.choices = choices
        super().__init__(seed)
    
    def generate(self) -> Any:
        """
        Generates a random parameter
        :return: The generated parameter
        """
        return np.random.choice(self.choices)
    
    @property
    def extremes(self) -> List[Any]:
        """
        Returns the minimum value, the maximum value, and an average example value
        :return: The minimum value, the maximum value, and an average example value
        """
        return [self.choices[0], self.choices[len(self.choices) // 2], self.choices[-1]]

class BooleanGenerator(RandomGenerator):
    """
    Returns a random boolean value
    """
    def generate(self) -> Any:
        """
        Generates a random parameter
        :return: The generated parameter
        """
        return np.random.choice([True, False])
    
    @property
    def extremes(self) -> List[Any]:
        """
        Returns the minimum value, the maximum value, and an average example value
        :return: The minimum value, the maximum value, and an average example value
        """
        return [False, False, True]

class GaussianGenerator(RandomGenerator):
    """
    Returns a number from a Gaussian distribution
    """
    def __init__(self, mean: float = 0, std: float = 1, seed: Optional[int] = None):
        """
        Initialize a new instance of the random generator class
        :param mean: The mean of the Gaussian distribution
        :param std: The standard deviation of the Gaussian distribution
        :param seed: The seed to use for the random number generator
        """
        self.mean = mean
        self.std = std
        super().__init__(seed)
    
    def generate(self) -> Any:
        """
        Generates a random parameter
        :return: The generated parameter
        """
        return np.random.normal(self.mean, self.std)
    
    @property
    def extremes(self) -> List[Any]:
        """
        Returns the minimum value, the maximum value, and an average example value
        :return: The minimum value, the maximum value, and an average example value
        """
        return [self.mean - self.std, self.mean, self.mean + self.std]

class PointGenerator(RandomGenerator):
    """
    Takes a list of tuples of the form (min, max) and returns a tuple of dimension len(min_max_list)
    """
    def __init__(self, min_max_list: List[Tuple[float, float]], seed: Optional[int] = None):
        """
        Initialize a new instance of the random generator class
        :param min_max_list: A list of tuples of the form (min, max)
        :param seed: The seed to use for the random number generator
        """
        self.min_max_list = min_max_list
        super().__init__(seed)
    
    def generate(self) -> Any:
        """
        Generates a random parameter
        :return: The generated parameter
        """
        return tuple(np.random.uniform(min_val, max_val) for min_val, max_val in self.min_max_list)
    
    @property
    def extremes(self) -> List[Any]:
        """
        Returns the minimum value, the maximum value, and an average example value
        :return: The minimum value, the maximum value, and an average example value
        """
        return [tuple(min_val for min_val, _ in self.min_max_list),
                tuple((min_val + max_val) / 2 for min_val, max_val in self.min_max_list),
                tuple(max_val for _, max_val in self.min_max_list)]

class Transformation(ABC):
    """
    An abstract class that defines the interface for all transformations
    """
    def __init__(self, seed: Optional[int] = None, override: Optional[int] = None):
        """
        Initialize a new instance of the transformation class
        :param seed: The seed to use for the random number generator
        :param override: Sets all values of the parameterization to the given value
        """
        # self.param_ids is an array of Union[str, Tuple[str, RandomGenerator]]
        # If an entry is of type str, we assume the Generator is the DefaultGenerator
        param_str_ids = []
        param_generators = []
        self.seed = seed
        for param_id in self.param_ids:
            if isinstance(param_id, str):
                param_str_ids.append(param_id)
                param_generators.append(DefaultGenerator(seed))
            else:
                param_str_ids.append(param_id[0])
                param_generators.append(param_id[1])
        self.param = Parameterization(param_str_ids, param_generators, seed, override)

    @abstractmethod
    def transform(self, input_data: Any, parameterization: Parameterization):
        """
        An abstract method that transforms the input data based on the parameterization
        :param input_data: The data to transform
        :param parameterization: The parameterization to use for the transformation
        :return: The transformed data
        """
        raise NotImplementedError

    def __call__(self, input_data):
        """
        Call method that transforms the input data based on the current parameterization
        :param input_data: The data to transform
        :return: The transformed data
        """
        return self.transform(input_data, self.param.get_parameterization())

    @property
    @abstractmethod
    def id(self):
        """
        :return: The unique identifier string for the transformation
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def param_ids(self):
        """
        :return: The parameterization size for the transformation
        """
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.id}({self.param})"

# class TransformationPipeline:
#     """
#     A utility to initialize a pipeline of transformations
#     """
#     def __init__(self, transformations: List[Transformation], seed: Optional[int] = None):
#         """
#         Initialize a new instance of the transformation pipeline class
#         :param transformations: The list of transformations to apply
#         :param seed: The seed to use for the random number generator
#         """
#         self.transformations = transformations
#         super().__init__(seed)

#     def transform(self, input_data: Any, parameterization: Parameterization):
#         """
#         Transforms the input data based on the parameterization
#         :param input_data: The data to transform
#         :param parameterization: The parameterization to use for the transformation
#         :return: The transformed data
#         """
#         for transformation in self.transformations:
#             input_data = transformation.transform(input_data, parameterization)
#         return input_data

#     @property
#     def id(self):
#         """
#         :return: The unique identifier string for the transformation
#         """
#         return 'pipeline'

#     @property
#     def param_ids(self):
#         """
#         :return: The parameterization size for the transformation
#         """
#         return [transformation.id for transformation in self.transformations]