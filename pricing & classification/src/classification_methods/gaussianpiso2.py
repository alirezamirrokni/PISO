from src.classification_methods.general_piso import ClassificationPISOBase


class GaussianPISO2(ClassificationPISOBase):
    name = "GaussianPISO2"
    variant = "gaussian"
    two_level = True
