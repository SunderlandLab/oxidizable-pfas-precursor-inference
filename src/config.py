from typing import List, Dict
import yaml
from dataclasses import dataclass
import pprint
pp = pprint.PrettyPrinter(indent=4, depth=2)


@dataclass
class Config:
    possible_precursors: List[str]
    prior_name: str
    jeffreys_variance: float
    ft_relativechain_yields: List[float]
    ft_relativechain_yield_errors: List[float]
    ecf_relativechain_yields: List[float]
    ecf_relativechain_yield_errors: List[float]
    specific_yield_overrides: Dict[str, float]
    specific_error_overrides: Dict[str, float]

    def print(self):
        print('Config object:')
        pp.pprint(self.__dict__)

    @classmethod
    def from_yaml(cls, filename):
        ym = yaml.safe_load(open(filename, 'r'))
        return Config(possible_precursors= ym.get('possible_precursors',[]),
                    prior_name=ym.get('prior_name','unknown'),
                    jeffreys_variance=ym.get('jeffreys_variance',None),
                    ft_relativechain_yields=ym.get('x_ft',[]),
                    ft_relativechain_yield_errors=ym.get('err_ft',[]),
                    ecf_relativechain_yields=ym.get('x_ecf',[]),
                    ecf_relativechain_yield_errors=ym.get('err_ecf',[]),
                    specific_yield_overrides=ym.get('specific_yields',{}),
                    specific_error_overrides=ym.get('specific_errors', {})
                    )

