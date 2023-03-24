from typing import List, Dict
from dataclasses import dataclass
import numpy as np

ALL_TARGETED = ['4:2 FT', '5:3 FT', '6:2 FT', '7:3 FT', '8:2 FT',
                '9:3 FT', '10:2 FT', 'C4 ECF', 'C5 ECF', 'C6 ECF',
                'C7 ECF', 'C8 ECF', 'C9 ECF', 'C10 ECF']
ALL_PFCA = [f'C{x}' for x in range(20)]
ALL_KEYS = ALL_PFCA + [f'{x}pre' for x in ALL_PFCA] + [f'{x}post' for x in ALL_PFCA] \
            + [f'{x}err' for x in ALL_PFCA] + [f'{x}MDL' for x in ALL_PFCA+ALL_TARGETED] \
            + ALL_TARGETED + [f'{x}meas' for x in ALL_TARGETED] \
            + ['PFOS', 'PFOSerr', 'PFOSpre', 'PFOSpost', 'PFOSMDL', 'CXerr']



class Measurement:
    value: float
    error: float
    MDL: float

    def __init__(self, error: float, MDL: float, value=None, post_value=None, pre_value=None):
        if (post_value is not None) and (pre_value is not None):
            self.value = max(0, post_value-pre_value)
            self.error = np.sqrt(2)*error
        elif (value is not None):
            self.value = value
            self.error = error
        else:
            raise ValueError("Value or pre and post need to be set")
        self.MDL = MDL

    def __str__(self):
        return f'Measurement(value={self.value}, error={self.error}, MDL={self.MDL}'

    def __repr__(self):
        return f'Measurement(value={self.value}, error={self.error}, MDL={self.MDL}'

@dataclass
class Measurements:
    PFCAs: Dict[str, Measurement]
    targeted_precursors: Dict[str, Measurement]
    PFOS: Measurement
    associated_config: str


    @classmethod
    def from_row(cls, dfrow):
        measured_values = {}
        pfcas_measured = {}
        targeted_precursors = {}
        PREPOST = False
        BLANKETERROR = False
        present_keys = [key for key in dfrow.keys() if key in ALL_KEYS]
        for key in present_keys:
            val = dfrow[key]
            if not np.isnan(val):
                measured_values[key] = val

        for chain in ALL_PFCA:
            this_pre = measured_values.get(f'{chain}pre',None)
            this_post = measured_values.get(f'{chain}post',None)
            this_val = measured_values.get(chain, None)
            if ((this_pre is not None) and (this_post is not None)) or (this_val is not None):
                this_MDL = measured_values.get(f'{chain}MDL',None)
                assert (this_MDL is not None),  f'{chain} MDL missing'

                this_err = measured_values.get(f'{chain}err', None)
                if this_err is None:
                    this_err = measured_values.get(f'CXerr',None)
                assert (this_err is not None), f'{chain} error or generic CXerr must be provided'

                pfcas_measured[chain] = Measurement(error=this_err,
                                                    MDL=this_MDL,
                                                    value=this_val,
                                                    post_value=this_post,
                                                    pre_value=this_pre)

        for targ in ALL_TARGETED:
            this_val = measured_values.get(targ, None)
            if this_val is not None:
                this_MDL = measured_values.get(f'{targ}MDL',None)
                assert (this_MDL is not None),  f'{targ} MDL missing'

                this_err = measured_values.get(f'{targ}err', None)
                if this_err is None:
                    this_err = measured_values.get(f'CXerr',None)
                assert (this_err is not None), f'{targ} error or generic CXerr must be provided'

                targeted_precursors[targ.replace('meas','')] = Measurement(error=this_err,
                                                    MDL=this_MDL,
                                                    value=this_val)

        try:
            cfg = 'config/'+dfrow['config']
        except KeyError:
            raise KeyError("Config file location must be provided.")

        PFOS = Measurement(error=measured_values.get('PFOSerr', measured_values.get('CXerr', None)),
                            MDL=measured_values.get('PFOSMDL', None),
                            value=measured_values.get('PFOS',None),
                            post_value=measured_values.get('PFOSpost',None),
                            pre_value=measured_values.get('PFOSpre',None))


        return Measurements(PFCAs=pfcas_measured, targeted_precursors=targeted_precursors,
                                PFOS=PFOS, associated_config=cfg)
