from typing import Dict, Tuple
from .config import Config

DEFAULT_CHAIN_FT = {0: 0.0245,
                    -1: 0.1850,
                    -2: 0.2874,
                    -3: 0.1943,
                    -4: 0.1425,
                    -5: 0.0867,
                    -6: 0.0}

DEFAULT_CHAIN_FT_ERR = {0: 0.0065,
                    -1: 0.0721,
                    -2: 0.0435,
                    -3: 0.0207,
                    -4: 0.0171,
                    -5: 0.0252,
                    -6: 0.0}

DEFAULT_CHAIN_ECF = {0: 0.0,
                    -1: 0.869,
                    -2: 0.0085,
                    -3: 0.0,}

DEFAULT_CHAIN_ECF_ERR = {0: 0.0,
                    -1: 0.1144,
                    -2: 0.0089,
                    -3: 0.0,}

C_EQUIVALENTS = {'4:2 FT': 4,
                '5:3 FT': 5, '6:2 FT': 6, '7:3 FT': 7, 
                '8:2 FT': 8, '9:3 FT': 9, '10:2 FT': 10,
                '11:3 FT': 11, '12:2 FT': 12, '13:3 FT': 13,
                '14:2 FT': 14,
                '5:3 FTCA': 5, '6:2 FTS': 6, '6:2 FTOH': 6, 
                '7:3 FTCA': 7, 
                '8:2 FTS': 8, '8:2 FTOH': 8, 
                'FOSA': 8,
                '9:3 FTCA': 9,
                '10:1 FTOH': 10,
                '10:2 FTS': 10, '10:2 FTOH': 10,
                '11:3 FTCA': 11, 
                '12:2 FTS': 12, '12:2 FTOH': 12,
                '13:1 FTOH': 13,
                '13:3 FTCA': 13,
                '14:2 FTS': 14, '14:2 FTOH': 14,
                'C4 ECF': 4, 'C5 ECF': 5, 'C6 ECF': 6,
                'C7 ECF': 7, 'C8 ECF': 8, 'C9 ECF': 9, 
                'C10 ECF': 10, 'C11 ECF': 11, 'C12 ECF': 12,
                'C13 ECF': 13, 'C14 ECF': 14, 'C15 ECF': 15}

class YieldGenerator:
    yield_lookup: Dict[str, float]
    error_lookup: Dict[str, float]

    def __init__(self, config: Config):
        precursors = config.possible_precursors
        self.yield_lookup = {}
        self.error_lookup = {}
        if len(config.ft_relativechain_yields) > 0:
            ft_length_yields = {-i: y for i,y in enumerate(config.ft_relativechain_yields)}
            ft_length_errors = {-i: y for i,y in enumerate(config.ft_relativechain_yield_errors)}
        else:
            ft_length_yields = DEFAULT_CHAIN_FT
            ft_length_errors = DEFAULT_CHAIN_FT_ERR
        if len(config.ecf_relativechain_yields) > 0:
            ecf_length_yields = {-i: y for i,y in enumerate(config.ecf_relativechain_yields)}
            ecf_length_errors = {-i: y for i,y in enumerate(config.ecf_relativechain_yield_errors)}
        for precursor in precursors:
            C = C_EQUIVALENTS[precursor]
            for C_prod in range(C,2,-1):
                rxn = f'{precursor}->C{C_prod}'
                defined = config.specific_yield_overrides.get(rxn, None)
                defined_err = config.specific_error_overrides.get(rxn, None)
                if defined is None:
                    if precursor.endswith('ECF'):
                        self.yield_lookup[rxn] = ecf_length_yields.get(C_prod - C, 0.0)
                        self.error_lookup[rxn] = ecf_length_errors.get(C_prod - C, 0.0)
                    else:
                        self.yield_lookup[rxn] = ft_length_yields.get(C_prod - C, 0.0)
                        self.error_lookup[rxn] = ft_length_errors.get(C_prod - C, 0.0)
                else:
                    self.yield_lookup[rxn] = defined
                    self.error_lookup[rxn] = defined_err
            

    def get_yield(self, precursor: str, pfca: str) -> Tuple[float,float]:
        return self.yield_lookup.get(f'{precursor}->{pfca}', 0.0), self.error_lookup.get(f'{precursor}->{pfca}', 0.0)
        
    