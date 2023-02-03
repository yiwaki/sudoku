# dftype Module
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.io.formats.style import Styler

PropertyName: TypeAlias = str
PropertyValue: TypeAlias = str
Property: TypeAlias = dict[PropertyName, PropertyValue]


def set_property_mask(
    arr: pd.DataFrame | Styler, mask: npt.NDArray[np.bool_], property: Property
) -> Styler:
    """set properties to DataFrame or Style of pandas

    Args:
        arr (DataFrame | Styler): pd.DataFrame or Styler of pandas
        mask (ndarray[np.bool_]): boolean array to set attributes
        property (Property): CSS property

    Raises:
        TypeError: if arr is not either DataFrame nor Styler

    Returns:
        Styler: Styler instance of Pandas
    """

    sty: Styler
    if isinstance(arr, pd.DataFrame):
        sty = arr.style
    elif isinstance(arr, Styler):
        sty = arr
    else:
        raise TypeError(f'unexpected data: {arr} {type(arr)}')

    for i in range(sty.data.shape[1]):  # type: ignore
        sty.set_properties(**property, subset=pd.IndexSlice[mask[:, i], i])  # type: ignore

    return sty


if __name__ == '__main__':
    import numpy as np

    arr = np.array([[0b1010, 0b1000], [0b0100, 0b1111]])
    df = pd.DataFrame(arr)
    msk = np.array([[True, False], [False, True]])
    sty = {'color': 'green'}

    sdf = set_property_mask(df, msk, sty)
    print(sdf.to_html())
