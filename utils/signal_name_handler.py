from enum import Enum


class LaneSide(Enum):
    """
    Lane boundary side
    """
    LH = 1  # Left
    RH = 2  # Right


class SignalName(Enum):
    """
    Coefficient value of line polynomial

    C0 : Lateral distance [m]
    C1 : Yaw angle [radian]
    C2 : Curvature/2 [1/m]
    C3 : Curvature rate/6 [1/m^2]
    ViewRangeEnd : Maximum longitudinal distance of view range
    ViewRangeStart : Minimum longitudinal distance of view range
    Confidence : Reliability of LD output
    """
    C0 = 1
    C1 = 2
    C2 = 3
    C3 = 4
    ViewRangeEnd = 5
    ViewRangeStart = 5.1
    Confidence = 6
    Quality = 6.1


class ConfusionMatrix(Enum):
    """
    Confusion matrix

    TP : true positive
    FN : false negative
    FP : false positive
    TN : true negative
    """
    TP = 0
    FN = 1
    FP = 2
    TN = 3


if __name__ == '__main__':
    side = LaneSide
    coeffi = SignalName
    for i in LaneSide:
        print(i.name)
        for i in coeffi:
            print(i.name, int(i.value))

    # CM = ConfusionMatrix
    # for i in CM:
    #     print(i.name)
    CM = ConfusionMatrix
    temp = [1, 2, 3, 4]
    print(temp[ConfusionMatrix.TP.value])