"""给你的 Python 迭代器加上魔法

一个 Python 迭代器的包装器，使其具有与Rust中的其他方法类似的风格，以提高迭代器操作的一致性和代码的可读性。
"""
import operator
import sys
from itertools import (
    accumulate,
    chain,
    combinations,
    combinations_with_replacement,
    compress,
    cycle,
    dropwhile,
    filterfalse,
    groupby,
    islice,
    repeat,
    starmap,
    takewhile,
    tee,
    zip_longest,
)
from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    NoReturn,
    Optional,
    Protocol,
    Reversible,
    Sequence,
    Sized,
    SupportsIndex,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

__all__ = ["ArkoWrapper"]

Wrapper = TypeVar("Wrapper", bound="ArkoWrapper")
T = TypeVar("T")
E = TypeVar("E")
C = TypeVar("C")
default_max = sys.maxsize


@runtime_checkable
class Searchable(Protocol[T]):
    def __iter__(self) -> Iterable[T]: ...

    def __getitem__(self, item) -> T: ...

    def __len__(self) -> int: ...


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
            max_operate_times: Optional[int] = default_max
    ) -> NoReturn:
        if isinstance(iterable, Iterable):
            self.__root__ = iterable
        elif iterable is None:
            self.__root__ = []
        else:
            self.__root__ = [iterable]

        if max_operate_times <= 0:
            raise ValueError(f"Requires a positive number: {max_operate_times}")
        elif max_operate_times > default_max:
            raise ValueError(f"'max_operate_times' cannot exceed {default_max}")
        self._max = max_operate_times

    def __str__(self) -> str:
        return str(self.__root__)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} " + "{" + f"{self.__root__}" + "}>"

    def _tee(self) -> Iterable[T]:
        """将已有迭代器分裂一次"""
        result, self.__root__ = tee(self.__root__)
        return result

    def _max_gen(self) -> Iterator:
        """将自己的迭代器现在某个范围内"""
        iter_values = iter(self._tee())
        try:
            for _ in range(self._max):
                yield next(iter_values)
        except StopIteration:
            pass

    def __add__(self: Wrapper, other: Union[E, Iterable[E]]) -> Wrapper:
        """实现加法操作。返回两个实列组成的新的迭代器的ArkoWrapper

        Args:
            other(Any): 相加的实列。

        Returns:
            ArkoWrapper: 返回两个实列组成的新的迭代器的ArkoWrapper
        """

        def generate() -> Iterator[Union[T, E]]:
            yield from self._tee()
            if not isinstance(other, str) and isinstance(other, Iterable):
                yield from other  # todo:复制生成器
            else:
                yield other

        return self.__class__(generate())

    def __radd__(self: Wrapper, other: Union[Iterable[E], E]) -> Wrapper:
        """实现反射加法操作。"""

        def generate() -> Iterator[Union[T, E]]:
            if not isinstance(other, str) and isinstance(other, Iterable):
                yield from other  # todo:复制生成器
            else:
                yield other
            yield from self._tee()

        return self.__class__(generate())

    def __eq__(self, other: Any) -> bool:
        """定义操作符(==)的行为。"""
        if isinstance(other, Sized) and self.length != len(other):
            return False
        if isinstance(other, Hashable) and self.__hash__() == hash(other):
            return True
        if isinstance(other, Iterable):
            for i in self.zip_longest(ArkoWrapper(other)):
                if i[0] != i[1]:
                    return False
            return True
        return self == [other]

    def __copy__(self: Wrapper) -> "Wrapper":
        return self.__class__(self.__root__)

    def __deepcopy__(self: Wrapper, *args) -> "Wrapper":
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
                    start=index.start, stop=index.stop, step=index.step
                )
            except ValueError:
                return self.__class__(list(self._max_gen()).__getitem__(index))
        try:
            if (index := int(index)) >= 0:
                target = self._tee()
            else:
                target = self.reverse()
                index = - index - 1
            iter_values = iter(target)
            time = -1
            while value := next(iter_values):
                if (time := time + 1) == index:
                    return value
        except StopIteration:
            raise ValueError(f"Out of range: {index}")
        except Exception:
            raise IndexError("Unsupported indexing for iterable")

    def __contains__(self, item: Any) -> bool:
        for elem in self._tee():
            if elem == item:
                return True
        return False

    def __hash__(self) -> int:
        return hash(self._tee())

    def __index__(self) -> int:
        """实现当对象用于切片表达式时到一个整数的类型转换。"""
        return self.__len__()

    def __iter__(self) -> Iterator[T]:
        """返回当前的迭代器。"""
        yield from self._tee()

    _len_cache: ClassVar[dict[int, int]] = {}

    def __len__(self) -> int:
        """返回当前的迭代器的长度，如果无限的话，则回返回最大操作次数。"""
        if (
                isinstance(self.__root__, Hashable)
                and
                (
                        (hash_value := hash(self.__root__))
                        in
                        self.__class__._len_cache
                )
        ):
            return self.__class__._len_cache[hash_value]
        if isinstance(self.__root__, Sized):
            length = len(list(self._tee()))
        else:
            length = 0
            for _ in self._tee():
                length += 1
                if length >= self._max:
                    return self._max
        if isinstance(self.__root__, Hashable):
            self.__class__._len_cache[hash(self.__root__)] = length
        return length

    def __matmul__(self, other: Any) -> T:
        """定义操作符(@)的行为。"""
        return self.__getitem__(other)

    def __mul__(self: Wrapper, times: Union[int, float, str]) -> "Wrapper":
        """实现乘法操作"""
        if (
                isinstance(times, SupportsIndex)
                and
                (times := int(float(times))) <= 0
        ):
            raise ValueError(f"'times' cannot be negative: {times}")
        try:
            return self.__class__(
                chain.from_iterable(repeat(tuple(self._tee()), int(times)))
            )
        except Exception:
            raise TypeError(f"Unsupported Type: {type(times)}.")

    def __neg__(self: Wrapper) -> "Wrapper":
        """定义取负操作"""
        return self.__reversed__()

    def __reversed__(self: Wrapper) -> "Wrapper":
        """定义反转"""
        if isinstance(self.__root__, Reversible):
            from copy import deepcopy as copy
            return self.__class__(reversed(copy(self.__root__)))
        else:
            return self.__class__(reversed(list(self._max_gen())))

    def __rshift__(
            self: Wrapper,
            target: Union[Callable[[Iterable[T], ...], C], Sequence]
    ) -> C:
        """实现右移位运算符 >>"""
        if isinstance(target, type) or callable(target):
            result: C = self.collect(target)
        elif isinstance(target, Sequence):
            # noinspection PyArgumentList,PyProtectedMember
            result: C = target.__class__(self.__radd__(target)._tee())
        else:
            raise TypeError(f"Unsupported value or type: '{target}'")
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
    def max_operate_time(self, max_operate_time: int) -> NoReturn:
        if max_operate_time <= 0:
            raise ValueError(f"Requires a positive number: {max_operate_time}")
        self._max = max_operate_time

    def accumulate(
            self: Wrapper, func: Optional[Callable[[T, T], C]] = operator.add,
            *,
            initial: Optional[int] = None
    ) -> Wrapper:
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

    def append(self: Wrapper, obj: E) -> Wrapper:
        """将对象附加到 ArkoWrapper 的末尾。注：此方法会修改本身的 __root__"""

        def clean() -> list[Union[E, T]]:
            temp = []
            iter_values = iter(self._tee())
            try:
                for _ in range(self._max):
                    temp.append(next(iter_values))
            except StopIteration:
                temp.append(obj)
                return temp

        result = self.__class__()
        self.__root__, result.__root__ = tee(clean())
        return self

    def chain(self: Wrapper, *iterables: Iterable[E]) -> Wrapper:
        """创建一个迭代器，它首先返回第一个可迭代对象中所有元素，接着返回下一个可迭代对象中所有元素，直到耗尽所有可迭代对象中的元素。"""
        return self.__class__(chain(self._tee(), *iterables))

    def collect(
            self,
            func: Optional[Callable[[Iterable[T], ...], C]] = list,
            *args, **kwargs
    ) -> C:
        """以整个迭代器作为参数，于给定的 func 函数中进行运算"""
        return func(self._tee(), *args, **kwargs)

    def combinations(self: Wrapper, r: int = 2) -> Wrapper:
        """返回由元素组成长度为 r 的子序列"""
        return self.__class__(combinations(self._tee(), r))

    def combinations_with_replacement(
            self: Wrapper, r: int = 2
    ) -> Wrapper:
        """返回由元素组成的长度为 r 的子序列，允许每个元素可重复出现。"""
        return self.__class__(combinations_with_replacement(self._tee(), r))

    def compress(self: Wrapper, selectors: Iterable) -> Wrapper:
        """返回元素中经 selectors 真值测试为 True 的元素"""
        return self.__class__(compress(self._tee(), selectors))

    def cycle(self: Wrapper) -> "Wrapper":
        return self.__class__(cycle(self._tee()))

    def drop_while(self: Wrapper, func: Callable) -> "Wrapper":
        return self.__class__(dropwhile(func, self._tee()))

    def empty(self) -> bool:
        return self.__len__() == 0

    def enumerate(self: Wrapper) -> Wrapper:
        def generator() -> Iterator[tuple[int, T]]:
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

    def extend(self: Wrapper, iterable: Iterable[E]) -> Wrapper:
        if not isinstance(iterable, Iterable):
            raise TypeError(f"{type(iterable)} object is not iterable")

        def generator() -> Iterator[Union[T, E]]:
            yield from self._tee()
            yield from iterable

        return self.__class__(generator())

    def fill(
            self: Wrapper,
            num: int,
            factory: Union[C, Callable[[], C]] = None
    ) -> Wrapper:
        if num > self._max:
            raise ValueError("'num' cannot exceed the maximum number of '_max'")
        elif num < 0:
            raise ValueError("'num' must be a positive number.")

        def generator() -> Iterator[C]:
            yield from self._tee()  # typed: T
            for _ in range(num):
                yield factory() if callable(factory) else factory  # typed: C

        return self.__class__(generator())

    def fill_to(
            self: Wrapper,
            num: int,
            factory: Union[C, Callable[[], C]] = None
    ) -> Wrapper:
        if num < self.length:
            raise ValueError("'num' cannot be less than its own length.")
        if num == (length := self.length):
            return self
        else:
            return self.fill(num - length, factory=factory)

    def filter(self: Wrapper, func: Callable[[T], Any]) -> Wrapper:
        return self.__class__(filter(func, self._tee()))

    def filter_false(self: Wrapper, func: Callable[[T], Any]) -> Wrapper:
        return self.__class__(filterfalse(func, self._tee()))

    def find(
            self, target: Any, *, full: Optional[bool] = False
    ) -> Iterator[int]:
        for t in self.enumerate():
            if t[1] == target:
                yield t[0]
                if not full:
                    break

    def flat(self: Wrapper, depth: int = -1) -> Wrapper:
        def generator(iterator: Iterable, times: int) -> Iterator[T]:
            for i in iterator:
                if isinstance(i, Iterable) and not isinstance(i, str) and times:
                    yield from generator(i, times - 1)
                else:
                    yield i

        return self.__class__(generator(self._tee(), depth))

    def group(self: Wrapper, n: int, fill_value: Any = None) -> "Wrapper":
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

    def map(
            self: Wrapper, func: Callable, start: Optional[int] = 0
    ) -> "Wrapper":
        def generator() -> Iterator[T]:
            iter_values = iter(self._tee())
            for _ in range(start):
                next(iter_values)
            yield from iter_values

        return self.__class__(map(func, generator()))

    def mutate(
            self: Wrapper, func: Callable[[Iterable[T], ...], Iterable] = list,
            *args, **kwargs
    ) -> "Wrapper":
        """改变 __root__ 的类型。"""
        self.__root__ = func(self.__root__, *args, **kwargs)
        return self

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

    def range(self: Wrapper, start: Optional[int] = 0) -> Wrapper:
        def generator() -> Iterator[T]:
            iter_values = iter(self._tee())
            try:
                for _ in range(start):
                    next(iter_values)
                index = start
                for _ in range(self.max_operate_time):
                    next(iter_values)
                    yield index
                    index += 1
            except StopIteration:
                ...

        return self.__class__(generator())

    def remove(self, target: Any) -> "ArkoWrapper":
        def generator() -> Iterator[T]:
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
            self: Wrapper, times: Optional[Union[int, float, str]] = None
    ) -> Wrapper:
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

    def reverse(self: Wrapper) -> "Wrapper":
        return self.__reversed__()

    @overload
    def slice(self: Wrapper, stop: int) -> "Wrapper":  # pragma: no cover
        ...

    @overload
    def slice(
            self: Wrapper, start: int, stop: int, step: Optional[int] = 1
    ) -> "Wrapper":  # pragma: no cover
        ...

    def slice(self: Wrapper, *args, **kwargs) -> "Wrapper":
        return self.__class__(islice(self._tee(), *args, *kwargs.values()))

    def search(
            self,
            sub: Searchable[E], *,
            func: Callable[[T, E], bool] = operator.eq
    ) -> Generator:
        target = self.tee()
        sub: ArkoWrapper = ArkoWrapper(sub)
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
            if func(target[i], sub[j]):
                j += 1
            if j == len(sub):
                yield i - j + 1
                j = partial[j - 1]

    def sort(
            self: Wrapper, key: Optional[Callable] = None, reverse: bool = False
    ) -> "Wrapper":
        return self.__class__(sorted(self._tee(), key=key, reverse=reverse))

    def starmap(self: Wrapper, func: Callable) -> "Wrapper":
        return self.__class__(starmap(func, self._tee()))

    def take_while(
            self, func: Union[Any, Callable[[T], bool]] = True
    ) -> Wrapper:
        if callable(func):
            return self.__class__(takewhile(func, self._tee()))
        elif bool(func):
            return self.__deepcopy__()
        else:
            return self.__class__()

    def tee(
            self: Wrapper, n: Optional[int] = None
    ) -> Union["Wrapper", Iterable["Wrapper"]]:
        if n is None:
            return self.__class__(self._tee())
        return (self.__class__(item) for item in tee(self.__root__, n))

    def unique(self: Wrapper, key: Optional[Callable] = None) -> "Wrapper":
        def generator() -> Iterator[T]:
            for k, g in groupby(sorted(self._tee(), key=key)):
                yield k

        return self.__class__(generator())

    def unwrap(
            self, func: Optional[Callable[[Iterable[T]], E]] = None
    ) -> Union[Iterable[T], E]:
        if func is None:
            # noinspection PyBroadException
            try:
                # noinspection PyArgumentList
                return self.__root__.__class__(self._tee())
            except Exception:
                return self._tee()
        else:
            return func(self._tee())

    if sys.version_info >= (3, 10):
        def zip(
                self: Wrapper, *iterables: Iterable[E],
                strict: Optional[bool] = False
        ) -> Wrapper:
            return self.__class__(zip(self._tee(), *iterables, strict=strict))
    else:
        def zip(self: Wrapper, *iterables: Iterable[E]) -> Wrapper:
            return self.__class__(zip(self._tee(), *iterables))

    def zip_longest(
            self: Wrapper, *iterables: Iterable[E],
            fill_value: Optional[C] = None
    ) -> Wrapper:
        return self.__class__(
            zip_longest(
                self.tee(), *ArkoWrapper(iterables).tee(), fillvalue=fill_value
            )
        )
