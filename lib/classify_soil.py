import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon

# scheme is (sand, clay, silt)
def ternary_2_cartesian(f_sand, f_clay, f_silt):
    """Convert ternary to cartesian coordinates"""
    x = (1 / 2) * (2 * f_silt + f_clay) / (f_sand + f_silt + f_clay)
    y = (np.sqrt(3) / 2) * (f_clay) / (f_sand + f_silt + f_clay)
    return (x, y)

p1 = (1.00, 0.00, 0.00)
p2 = (0.90, 0.10, 0.00)
p3 = (0.85, 0.00, 0.15)

tern_poly = [p1, p2, p3]
sand_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

p1 = (0.90, 0.10, 0.00)
p2 = (0.85, 0.15, 0.00)
p3 = (0.70, 0.00, 0.30)
p4 = (0.85, 0.00, 0.15)

tern_poly = [p1, p2, p3, p4]
loamy_sand_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

p1 = (0.85, 0.15,  0.00)
p2 = (0.80, 0.20,  0.00)
p3 = (0.525, 0.20,  0.275)
p4 = (0.525, 0.075, 0.400)
p5 = (0.425, 0.075, 0.500)
p6 = (0.500, 0.000, 0.500)
p7 = (0.70, 0.00, 0.30)

tern_poly = [p1, p2, p3, p4, p5, p6, p7]
sandy_loam_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

p1 = (0.80, 0.20, 0.00)
p2 = (0.65, 0.35, 0.00)
p3 = (0.45, 0.35, 0.20)
p4 = (0.450, 0.275, 0.275)
p5 = (0.525, 0.200, 0.275)

tern_poly = [p1, p2, p3, p4, p5]
sandy_clay_loam_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

p1 = (0.65, 0.35, 0.00)
p2 = (0.45, 0.55, 0.00)
p3 = (0.45, 0.35, 0.20)

tern_poly = [p1, p2, p3]
sandy_clay_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

p1 = (0.45, 0.55, 0.00)
p2 = (0.00, 1.00, 0.00)
p3 = (0.00, 0.60, 0.40)
p4 = (0.20, 0.40, 0.40)
p5 = (0.45, 0.40, 0.15)

tern_poly = [p1, p2, p3, p4, p5]
clay_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

p1 = (0.20, 0.40, 0.40)
p2 = (0.00, 0.60, 0.40)
p3 = (0.00, 0.40, 0.60)

tern_poly = [p1, p2, p3]
silty_clay_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

p1 = (0.45, 0.40, 0.15)
p2 = (0.20, 0.40, 0.40)
p3 = (0.200, 0.275, 0.525)
p4 = (0.450, 0.275, 0.275)

tern_poly = [p1, p2, p3, p4]
clay_loam_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

p1 = (0.525, 0.200, 0.275)
p2 = (0.450, 0.275, 0.275)
p3 = (0.225, 0.275, 0.500)
p4 = (0.425, 0.075, 0.50)
p5 = (0.525, 0.075, 0.40)

tern_poly = [p1, p2, p3, p4, p5]
loam_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

p1 = (0.200, 0.275, 0.525)
p2 = (0.200, 0.40, 0.40)
p3 = (0.000, 0.400, 0.600)
p4 = (0.000, 0.275, 0.725)

tern_poly = [p1, p2, p3, p4]
silty_clay_loam_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

p1 = (0.500, 0.000, 0.500)
p2 = (0.225, 0.275, 0.500)
p3 = (0.000, 0.275, 0.725)
p4 = (0.000, 0.125, 0.875)
p5 = (0.075, 0.125, 0.800)
p6 = (0.20, 0.000, 0.800)

tern_poly = [p1, p2, p3, p4, p5, p6]
silt_loam_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

p1 = (0.200, 0.000, 0.800)
p2 = (0.075, 0.125, 0.800)
p3 = (0.000, 0.125, 0.875)
p4 = (0.000, 0.000, 1.000)

tern_poly = [p1, p2, p3, p4]
silt_cart = Polygon([ternary_2_cartesian(*sp) for sp in tern_poly])

usda_classification = {
    'sand'            : sand_cart,
    'loamy sand'      : loamy_sand_cart,
    'sandy loam'      : sandy_loam_cart,
    'sandy clay loam' : sandy_clay_loam_cart,
    'sandy clay'      : sandy_clay_cart,
    'clay'            : clay_cart,
    'silty clay'      : silty_clay_cart,
    'clay loam'       : clay_loam_cart,
    'loam'            : loam_cart,
    'silty clay loam' : silty_clay_loam_cart,
    'silt loam'       : silt_loam_cart,
    'silt'            : silt_cart
}


def classify_soil(f_sand, f_clay, f_silt, classifier=usda_classification):
    """Classify soil given a particular classification scheme"""
    x, y = ternary_2_cartesian(f_sand, f_clay, f_silt)
    p = Point((x,y))
    for name, classification in classifier.items():
        if p.within(classification):
            return name.upper()
    return None
