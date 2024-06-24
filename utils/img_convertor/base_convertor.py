from abc import ABC, abstractmethod

class BaseConvertor(ABC):
    @abstractmethod
    def convert(self, image):
        pass

    @abstractmethod
    def convert(self, image, output_path):
        pass