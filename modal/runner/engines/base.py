from abc import ABC, abstractmethod
from typing import Any, Generator


class BaseEngine(ABC):
    # @abstractmethod
    # async def tokenize_prompt(self) -> List[int]:
    #     pass

    # @abstractmethod
    # async def max_model_len(self) -> int:
    #     pass

    @abstractmethod
    async def generate(self) -> Generator[str, Any, None]:
        pass
