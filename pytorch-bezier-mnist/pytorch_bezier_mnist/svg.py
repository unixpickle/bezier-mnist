import base64
from typing import Any, List, Tuple


def beziers_to_svg(loops: List[List[List[Tuple[float, float]]]]) -> str:
    """
    Encode a collection of Bezier curves into an SVG image.

    :return: the contents of an SVG file.
    """
    path = ""
    for loop in loops:
        loop_data = f"M %f,%f" % loop[0][0]
        for curve in loop:
            loop_data += "C %f,%f %f,%f %f,%f " % (*curve[1], *curve[2], *curve[3])
        loop_data += "Z"
        path += loop_data
    return (
        '<?xml version="1.0" encoding="utf-8" ?>'
        + '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28">'
        + '<rect x="0" y="0" width="28" height="28" fill="black" />'
        + '<path fill="white" d="'
        + path
        + '" />'
        + "</svg>"
    )


def beziers_to_ipython_image(loops: List[List[List[Tuple[float, float]]]]) -> Any:
    """
    Encode the Bezier curves into an image suitable to be displayed in an
    IPython/Jupyter notebook.
    """
    from IPython.display import SVG

    data = beziers_to_svg(loops)
    return SVG(
        url="data:image/svg+xml;base64,"
        + str(base64.b64encode(bytes(data, "utf-8")), "utf-8")
    )
