import abc


class BaseExtractor(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def extract(self, image):
        """Extracts features from the given image using a specific class-dependent technique."""
        return
