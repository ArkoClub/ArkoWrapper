"""给你的 Python 迭代器加上魔法

一个 Python 迭代器的包装器，使其具有与Rust中的其他方法类似的风格，以提高迭代器操作的一致性和代码的可读性。
"""
import operator
import sys
from collections.abc import (
    Iterable,
    Sized,
    Callable,
    Generator,
)
from itertools import (
    chain,
    tee,
    repeat,
    accumulate,
    combinations,
    combinations_with_replacement,
    cycle,
    dropwhile,
    filterfalse,
    zip_longest,
    islice,
    compress,
    groupby,
    starmap,
)
from typing import (
    Any,
    Generic,
    overload,
    Optional,
    Union,
    NoReturn,
    TypeVar,
    Reversible,
    Sequence,
)

__all__ = ["ArkoWrapper"]

M = TypeVar("M")
T = TypeVar("T")
E = TypeVar("E")
C = TypeVar("C")


class ArkoWrapper(Generic[T]):
    """一个 Python 迭代器的包装器

    Attributes:
        iterable: 需要被 Wrap 的 Object。 可以是一个迭代器或者其它任何Object。
        max_operate_times: Wrapper 操作次数的上限。用于限制无限的迭代器。
    """
    __root__: Iterable[T]
    _max: int

    __slots__ = '__root__', '_max'

    # noinspection PyTypeChecker
    def __init__(
            self, iterable: Optional[Union[Iterable[T], T]] = None, *,
            max_operate_times: Optional[int] = sys.maxsize
    ) -> NoReturn:
        if isinstance(iterable, Iterable):
            self.__root__ = iterable
        elif iterable is None:
            self.__root__ = []
        else:
            self.__root__ = [iterable]

        if max_operate_times <= 0:
            raise ValueError(f"Requires a positive number: {max_operate_times}")
        self._max = max_operate_times

    def __str__(self) -> str:
        return str(self.__root__)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.__root__}>"

    def _tee(self) -> Iterable[T]:
        """将已有迭代器分裂一次"""
        result, self.__root__ = tee(self.__root__)
        return result

    def _max_gen(self) -> Generator[T]:
        """将自己的迭代器现在某个范围内"""
        iter_values = iter(self._tee())
        try:
            for _ in range(self._max):
                yield next(iter_values)
        except StopIteration:
            ...

    def __add__(self: M, other: E) -> "M[Union[T, E]]":
        """实现加法操作。返回两个实列组成的新的迭代器的ArkoWrapper

        Args:
            other(Any): 相加的实列。

        Returns:
            ArkoWrapper: 返回两个实列组成的新的迭代器的ArkoWrapper
        """

        def generate() -> Generator:
            yield from self._tee()
            if isinstance(other, ArkoWrapper):
                yield from other._tee()
            elif isinstance(other, Iterable):
                yield from tee(other)[0]
            else:
                yield other

        return self.__class__(generate())

    def __radd__(self: M, other: E) -> "M[Union[T, E]]":
        """实现反射加法操作。"""

        def generate() -> Generator:
            if isinstance(other, ArkoWrapper):
                yield from other._tee()
            elif isinstance(other, Iterable):
                yield from tee(other)[0]
            else:
                yield other
            yield from self._tee()

        return self.__class__(generate())

    def __eq__(self, other: Any) -> bool:
        """定义操作符(==)的行为。"""
        if any([
            isinstance(other, ArkoWrapper) and self.__root__ == other.__root__,
            isinstance(other, Iterable) and self.__root__ == other
        ]):
            return True
        if isinstance(other, (ArkoWrapper, Iterable)):
            for i in self.zip(other):
                if i[0] != i[1]:
                    return False
            return True
        return self == [other]

    def __copy__(self: M) -> "M":
        """定义对类的实例使用 copy.copy() 时的行为"""
        return self.__class__(self._tee())

    def __getitem__(self, index: Any) -> T:
        """定义对容器中某一项使用 self[key] 的方式进行读取操作时的行为"""
        if isinstance(index, slice):
            if '__getitem__' in dir(self.__root__):
                # noinspection PyUnresolvedReferences
                return self.__root__[index]
            try:
                return self.slice(
                    *filter(None, [index.start, index.step, index.stop])
                )
            except ValueError:
                return list(self._max_gen()).__getitem__(index)
        try:
            if (index := int(index)) > 0:
                target = self._tee()
            else:
                target = self.reverse()
                index = - index - 1
            iter_values = iter(target)
            time = -1
            while value := next(iter_values):
                try:
                    if (time := time + 1) == index:
                        return value
                except StopIteration:
                    raise ValueError(f"Out of range: {index}")
        except Exception:
            raise IndexError("Unsupported indexing for iterable")

    def __index__(self) -> int:
        """实现当对象用于切片表达式时到一个整数的类型转换。"""
        return self.__len__()

    def __iter__(self) -> T:
        """返回当前的迭代器。"""
        return self._tee().__iter__()

    def __len__(self) -> int:
        """返回当前的迭代器的长度，如果无限的话，则回返回最大操作次数。"""
        if isinstance(self.__root__, Sized):
            return len(list(self._tee()))
        else:
            length = 0
            for _ in self._tee():
                length += 1
                if length >= self._max:
                    return self._max
            return length

    def __matmul__(self, other: Any) -> T:
        """定义操作符(@)的行为。"""
        return self.__getitem__(other)

    def __mul__(self: M, times: Union[int, float, str]) -> "M":
        """实现乘法操作"""
        try:
            if (times := int(float(times))) <= 0:
                raise ValueError(f"'times' cannot be negative: {times}")
            return self.__class__(
                chain.from_iterable(repeat(tuple(self._tee()), times))
            )
        except (ValueError, TypeError):
            raise TypeError(f"Unsupported Type: {type(times)}.")

    def __neg__(self: M) -> "M":
        """定义取负操作"""
        return self.__reversed__()

    def __reversed__(self: M) -> "M":
        """定义反转"""
        if isinstance(self.__root__, Reversible):
            from copy import deepcopy as copy
            return self.__class__(reversed(copy(self.__root__)))
        else:
            try:
                return self.__class__(reversed(list(self._max_gen())))
            except (ValueError, TypeError):
                raise TypeError(
                    f"The iter '{self.__root__}' is not 'Reversible.'")

    def __rshift__(self: M, target: Any) -> C:
        """实现右移位运算符 >>"""
        if isinstance(target, type) or callable(target):
            result: C = self.collect(target)
        elif isinstance(target, Sequence):
            result: C = self.collect(type(target))
        else:
            raise ValueError(f"Unsupported value or type: '{target}'")
        return result

    @property
    def root(self) -> Iterable[T]:
        return self.__root__

    @property
    def length(self) -> int:
        return self.__len__()

    @property
    def max_operate_time(self) -> int:
        """最大操作次数，防止无线递归"""
        return self._max

    @max_operate_time.setter
    def max_operate_time(self, max_operate_time: int):
        if max_operate_time <= 0:
            raise ValueError(f"Requires a positive number: {max_operate_time}")
        self._max = max_operate_time

    def accumulate(
            self: M, func: Optional[Callable[[T, T], C]] = operator.add, *,
            initial: Optional[int] = None
    ) -> "M[C]":
        return self.__class__(accumulate(self._tee(), func, initial=initial))

    def all(self) -> bool:
        """如果所有元素均为真值（或root为空）则返回 True"""
        iter_values = iter(self._tee())
        try:
            while bool(next(iter_values)):
                ...
            return False
        except StopIteration:
            return True

    def any(self) -> bool:
        """如果任一元素为真值则返回 True。 若root为空，返回 False"""
        iter_values = iter(self._tee())
        try:
            while not bool(next(iter_values)):
                ...
            return True
        except StopIteration:
            return False

    def chain(self: M, *iterables: Iterable[E]) -> "M[Union[T, E]]":
        """创建一个迭代器，它首先返回第一个可迭代对象中所有元素，接着返回下一个可迭代对象中所有元素，直到耗尽所有可迭代对象中的元素。"""
        return self.__class__(chain(self._tee(), *iterables))

    def collect(
            self,
            func: Optional[Callable[[Iterable[T], ...], C]] = list,
            *args, **kwargs
    ) -> C:
        """以整个迭代器作为参数，于给定的 func 函数中进行运算"""
        return func(self._tee(), *args, **kwargs)

    def combinations(self: M, r: int = 2) -> "M[combinations[tuple[T, T]]]":
        return self.__class__(combinations(self._tee(), r))

    def combinations_with_replacement(self: M, r: int = 2) -> "M":
        return self.__class__(combinations_with_replacement(self._tee(), r))

    def compress(self: M, selectors: Iterable) -> "M":
        return self.__class__(compress(self._tee(), selectors))

    def cycle(self: M) -> "M":
        return self.__class__(cycle(self._tee()))

    def drop_while(self: M, func: Callable) -> "M":
        return self.__class__(dropwhile(func, self._tee()))

    def enumerate(self: M) -> "M[tuple[int, T]]":
        def generator() -> Generator[tuple[int, T]]:
            iter_values = iter(self._tee())
            index = 0
            for _ in range(self.max_operate_time):
                try:
                    value = next(iter_values)
                    yield index, value
                    index += 1
                except StopIteration:
                    break

        return self.__class__(generator())

    def filter(self: M, func: Callable[[T], Any]) -> "M[T]":
        return self.__class__(filter(func, self._tee()))

    def filter_false(self: M, func: Callable[[T], Any]) -> "M[T]":
        return self.__class__(filterfalse(func, self._tee()))

    def find(
            self, target: Any, *, full: Optional[bool] = False
    ) -> Generator[int]:
        for t in self.enumerate():
            if t[1] == target:
                yield t[0]
                if not full:
                    break

    def flat(self: M, depth: int = -1) -> "M":
        def generate(iterator: Iterable, times: int) -> Generator:
            for i in iterator:
                if isinstance(i, Iterable) and not isinstance(i, str) and times:
                    yield from generate(i, times - 1)
                else:
                    yield i

        return self.__class__(generate(self._tee(), depth))

    def group(self: M, n: int, fill_value: Any = None) -> "M":
        iter_values = [iter(self._tee())] * n
        return self.__class__(zip_longest(*iter_values, fillvalue=fill_value))

    def join(self, sep: str = ', ') -> str:
        iter_values = iter(self._tee())
        result = ''
        try:
            for _ in range(self.max_operate_time):
                value = next(iter_values)
                result += f"{sep}{value}"
        except StopIteration:
            ...
        return result

    def map(self: M, func: Callable) -> "M":
        return self.__class__(map(func, self._tee()))

    def mutate(
            self: M, func: Callable[[Iterable[T], ...], Iterable] = list,
            *args, **kwargs
    ) -> "M":
        return self.__class__(func(self._tee(), *args, **kwargs))

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

    def range(self: M, start: Optional[int] = 0) -> "M":
        def generator() -> Generator[int]:
            iter_values = iter(self._tee())
            index = start
            try:
                while index > 0:
                    next(iter_values)
                    index -= 1
                index = start
                for _ in range(self.max_operate_time):
                    next(iter_values)
                    yield index
                    index += 1
            except StopIteration:
                return

        return self.__class__(generator())

    def remove(self, target: Any) -> "ArkoWrapper":
        def generator():
            iter_values = iter(self._tee())
            is_sequence = isinstance(target, Sequence)
            removed = False
            try:
                for _ in range(self.max_operate_time):
                    value = next(iter_values)
                    if (
                            (is_sequence and value in target)
                            or
                            (not removed and value == target)
                    ):
                        continue
                    else:
                        yield value
            except StopIteration:
                ...

        return ArkoWrapper(generator())

    def repeat(
            self: M, times: Optional[Union[int, float, str]] = None
    ) -> "M[T]":
        if times is None:
            return self.__class__(
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
            return self.__class__(
                chain.from_iterable(repeat(tuple(self._tee()), time + 1))
            )
        else:
            raise TypeError(f"Unsupported Type: {type(times)}.")

    def reverse(self: M) -> "M":
        return self.__reversed__()

    @overload
    def slice(self: M, stop: int) -> "M":
        ...

    @overload
    def slice(
            self: M, start: int, stop: int, step: Optional[int] = 1
    ) -> "M":
        ...

    def slice(self: M, *args) -> "M":
        return self.__class__(islice(self._tee(), *args))

    def search(self, sub: Sized) -> Generator[int]:
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
                yield i - j + 1
                j = partial[j - 1]

    def sort(
            self: M, key: Optional[Callable] = None, reverse: bool = False
    ) -> "M":
        return self.__class__(sorted(self._tee(), key=key, reverse=reverse))

    def starmap(self: M, func: Callable) -> "M":
        return self.__class__(starmap(func, self._tee()))

    def tee(
            self: M, n: Optional[int] = None
    ) -> Union["M", Generator["M"]]:
        if n is None:
            return self.__class__(self._tee())
        return (self.__class__(item) for item in tee(self.__root__, n))

    def unique(self: M) -> "M":
        def generator() -> Generator:
            for k, g in groupby(self._tee()):
                yield k

        return self.__class__(generator())

    def unwrap(self) -> Iterable[T]:
        return self._tee()

    def zip(
            self: M, *iterables: Iterable[E], strict: Optional[bool] = False
    ) -> "M[Union[T, E]]":
        if sys.version_info >= (3, 10):
            return self.__class__(zip(self._tee(), *iterables, strict=strict))
        return self.__class__(zip(self._tee(), *iterables))
