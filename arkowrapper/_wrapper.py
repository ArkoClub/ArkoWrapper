from collections.abc import (
    Iterable,
    Sized,
    Callable,
    Generator,
)
from itertools import *
from typing import (
    Any,
    overload,
    Optional,
    Union,
    NoReturn,
    TypeVar,
    Reversible,
)

__all__ = ["ArkoWrapper"]

T = TypeVar("T")


class ArkoWrapper(object):
    __root__: Iterable[Any]
    _max: int

    __slots__ = '__root__', '_max'

    # noinspection PyTypeChecker
    def __init__(
            self, iterable: Optional[Any] = None, *,
            max_num: Optional[int] = 2 ** 20
    ) -> NoReturn:
        if isinstance(iterable, Iterable):
            self.__root__ = iterable
        elif iterable is None:
            self.__root__ = []
        else:
            self.__root__ = [iterable]

        if max_num <= 0:
            raise ValueError(f"Requires a positive number: {max_num}")
        self._max = max_num

    def __str__(self) -> str:
        return str(self.__root__)

    def __repr__(self) -> str:
        return f"<ArkoWrapper:{self.__root__}>"

    def _tee(self) -> Iterable:
        result, self.__root__ = tee(self.__root__)
        return result

    def __add__(self, other: Any) -> "ArkoWrapper":
        def generate() -> Generator:
            yield from self._tee()
            if isinstance(other, ArkoWrapper):
                yield from other._tee()
            elif isinstance(other, Iterable):
                yield from tee(other)[0]
            else:
                yield other

        return ArkoWrapper(generate())

    def __radd__(self, other: Any) -> "ArkoWrapper":
        return self.__add__(other)

    def __copy__(self) -> "ArkoWrapper":
        return ArkoWrapper(self._tee())

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            if '__getitem__' in dir(self.__root__):
                # noinspection PyUnresolvedReferences
                return self.__root__[index]
            else:
                if any([
                    int(index.start) < 0,
                    int(index.step) < 0,
                    int(index.stop) < 0
                ]):
                    raise ValueError(
                        "Unsupported slicing conversion for iterable")
                return self.slice(index.start, index.step, index.stop)

        else:
            try:
                # noinspection PyUnresolvedReferences
                return self._tee()[index]
            except (KeyError, TypeError):
                if isinstance(index, int):
                    iter_values = iter(self._tee())
                    time = 0
                    while True:
                        try:
                            value = next(iter_values)
                            if time == index:
                                return value
                            time += 1
                        except StopIteration:
                            raise ValueError(f"Out of range: {index}")
                else:
                    raise IndexError("Unsupported indexing for iterable")

    def __index__(self) -> int:
        return self.__len__()

    def __iter__(self) -> Any:
        return self._tee().__iter__()

    def __len__(self) -> int:
        if isinstance(self.__root__, Sized):
            return len(list(self._tee()))
        else:
            length = 0
            for _ in self._tee():
                length += 1
                if length >= self._max:
                    return self._max
            return length

    def __matmul__(self, other: Any) -> Any:
        return self.__getitem__(other)

    def __mul__(self, times: Union[int, float, str]) -> "ArkoWrapper":
        if any([
            isinstance(times, (int, float)),
            isinstance(times, str) and times.isnumeric()
        ]):
            if (times := int(times)) <= 0:
                raise ValueError(f"'times' cannot be negative: {times}")
            return ArkoWrapper(
                chain.from_iterable(repeat(tuple(self._tee()), times))
            )
        else:
            raise TypeError(f"Unsupported Type: {type(times)}.")

    def __neg__(self) -> "ArkoWrapper":
        return self.__reversed__()

    def __reversed__(self) -> "ArkoWrapper":
        if isinstance(self.__root__, Reversible):
            # noinspection PyTypeChecker
            return ArkoWrapper(reversed(self._tee()))
        else:
            raise TypeError(f"The iter '{self.__root__}' is not 'Reversible.'")

    @property
    def root(self) -> Iterable[Any]:
        return self.__root__

    @property
    def length(self) -> int:
        return self.__len__()

    @property
    def max(self) -> int:
        return self._max

    @max.setter
    def max(self, max_num: int):
        if max_num <= 0:
            raise ValueError(f"Requires a positive number: {max_num}")
        self._max = max_num

    def accumulate(
            self, func: Callable, *, initial: Optional[int] = None
    ) -> "ArkoWrapper":
        return ArkoWrapper(accumulate(self._tee(), func, initial=initial))

    def chain(self, *iterables: Iterable) -> "ArkoWrapper":
        return ArkoWrapper(chain(self._tee(), *iterables))

    def collect(
            self, func: Callable[[Iterable, ...], T], *args, **kwargs
    ) -> T:
        return func(self._tee(), *args, **kwargs)

    def combinations(self, r: int = 2) -> "ArkoWrapper":
        return ArkoWrapper(combinations(self._tee(), r))

    def combinations_with_replacement(self, r: int = 2) -> "ArkoWrapper":
        return ArkoWrapper(combinations_with_replacement(self._tee(), r))

    def compress(self, selectors: Iterable) -> "ArkoWrapper":
        return ArkoWrapper(compress(self._tee(), selectors))

    def cycle(self) -> "ArkoWrapper":
        return ArkoWrapper(cycle(self._tee()))

    def drop_while(self, func: Callable) -> "ArkoWrapper":
        return ArkoWrapper(dropwhile(func, self._tee()))

    def enumerate(self) -> "ArkoWrapper":
        def generator() -> Generator[tuple[int, Any]]:
            iter_values = iter(self._tee())
            index = 0
            while True:
                try:
                    value = next(iter_values)
                    yield index, value
                    index += 1
                except StopIteration:
                    break

        return ArkoWrapper(generator())

    def filter(self, func: Callable[[...], Any]) -> "ArkoWrapper":
        return ArkoWrapper(filter(func, self._tee()))

    def filter_false(self, func: Callable[[...], Any]) -> "ArkoWrapper":
        return ArkoWrapper(filterfalse(func, self._tee()))

    def find(
            self, target: Any, *, full: Optional[bool] = False
    ) -> Generator[int]:
        for t in self.enumerate():
            if t[1] == target:
                yield t[0]
                if not full:
                    break

    def flat(self, depth: int = -1) -> "ArkoWrapper":
        def generate(iterator: Iterable, times: int) -> Generator:
            for i in iterator:
                if isinstance(i, Iterable) and not isinstance(i, str) and times:
                    yield from generate(i, times - 1)
                else:
                    yield i

        return ArkoWrapper(generate(self._tee(), depth))

    def group(self, n: int, fill_value: Any = None) -> "ArkoWrapper":
        iter_values = [iter(self._tee())] * n
        return ArkoWrapper(zip_longest(*iter_values, fillvalue=fill_value))

    def map(self, func: Callable) -> "ArkoWrapper":
        return ArkoWrapper(map(func, self._tee()))

    def mutate(
            self, func: Callable[..., Iterable] = list, *args, **kwargs
    ) -> "ArkoWrapper":
        return ArkoWrapper(func(self._tee(), *args, **kwargs))

    def print(
            self,
            length: Optional[int] = None, *,
            end: Optional[str] = ', ',
            print_func: Optional[Callable] = print
    ) -> "ArkoWrapper":
        time = 0
        iter_values = iter(self._tee())
        try:
            if length is not None:
                while length > 0:
                    print_func(next(iter_values), end=end)
                    length = length - 1
                    time += 1
            else:
                while time < self._max:
                    print_func(next(iter_values), end=end)
                    time += 1
        except StopIteration:
            pass
        if time != 0:
            print_func(len(end) * '\b')
        return self

    def range(self, start: Optional[int] = 0) -> "ArkoWrapper":
        def generator() -> Generator[int]:
            iter_values = iter(self._tee())
            index = start
            try:
                while index > 0:
                    next(iter_values)
                    index -= 1
                index = start
                while True:
                    next(iter_values)
                    yield index
                    index += 1
            except StopIteration:
                return

        return ArkoWrapper(generator())

    def repeat(
            self, times: Optional[Union[int, float, str]] = None
    ) -> "ArkoWrapper":
        if times is None:
            return ArkoWrapper(
                chain.from_iterable(
                    repeat(tuple(self._tee()), self._max + 1)
                )
            )
        if any([
            isinstance(times, str) and times.isnumeric(),
            isinstance(times, int)
        ]):
            if (time := int(times)) <= 0:
                raise ValueError(f"'times' cannot be negative: {times}")
            return ArkoWrapper(
                chain.from_iterable(repeat(tuple(self._tee()), time + 1))
            )
        else:
            raise TypeError(f"Unsupported Type: {type(times)}.")

    def reverse(self) -> "ArkoWrapper":
        return self.__reversed__()

    @overload
    def slice(self, stop: int) -> "ArkoWrapper":
        ...

    @overload
    def slice(
            self, start: int, stop: int, step: Optional[int] = 1
    ) -> "ArkoWrapper":
        ...

    def slice(self, *args) -> "ArkoWrapper":
        return ArkoWrapper(islice(self._tee(), *args))

    def search(self, sub: Sized) -> Iterable[int]:
        target = self.tee()
        sub = ArkoWrapper(sub)
        partial: list[int] = [0]
        for i in sub.range(1):
            j = partial[i - 1]
            while j > 0 and sub[j] != sub[i]:
                j = partial[j - 1]
            partial.append(j + 1 if sub[j] == sub[i] else j)

        j = 0

        for i in target.range():
            while j > 0 and target[i] != sub[j]:
                j = partial[j - 1]
            if target[i] == sub[j]:
                j += 1
            if j == len(sub):
                yield i - (j - 1)
                j = partial[j - 1]

    def sort(
            self, key: Optional[Callable] = None, reverse: bool = False
    ) -> "ArkoWrapper":
        return ArkoWrapper(sorted(self._tee(), key=key, reverse=reverse))

    def starmap(self, func: Callable) -> "ArkoWrapper":
        return ArkoWrapper(starmap(func, self._tee()))

    def tee(
            self, n: Optional[int] = None
    ) -> Union["ArkoWrapper", Generator["ArkoWrapper"]]:
        if n is None:
            return ArkoWrapper(self._tee())
        return (ArkoWrapper(item) for item in tee(self.__root__, n))

    def unique(self) -> "ArkoWrapper":
        def generator() -> Generator:
            for k, g in groupby(self._tee()):
                yield k

        return ArkoWrapper(generator())

    def unwrap(self) -> Iterable:
        return self._tee()

    def zip(self, *iterables, strict: bool = False) -> "ArkoWrapper":
        return ArkoWrapper(zip(self._tee(), *iterables, strict=strict))
