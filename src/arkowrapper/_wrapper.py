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
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    NoReturn,
    Optional,
    Reversible,
    Sequence,
    Sized,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import (
    Protocol,
    Self,
    SupportsIndex,
    runtime_checkable,
)

__all__ = ["ArkoWrapper"]

T = TypeVar("T")
E = TypeVar("E")
R = TypeVar("R")
default_max = sys.maxsize
NOT_SET = object()


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

    def __add__(self, other: Union[E, Iterable[E]]) -> Self:
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

    def __radd__(self, other: Union[Iterable[E], E]) -> Self:
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

    def __copy__(self) -> Self:
        return self.__class__(self.__root__)

    def __deepcopy__(self, *args) -> Self:
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
            index = int(index)
            if index >= 0:
                target = self._tee()
            else:
                target = self.reverse()
                index = - index - 1
            iter_values = iter(target)
            time = -1
            value = next(iter_values)
            while value:
                time += 1
                if time == index:
                    return value
                value = next(iter_values)
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

    if sys.version_info <= (3, 8):
        def __int__(self) -> int:
            return self.__index__()

    def __iter__(self) -> Iterator[T]:
        """返回当前的迭代器。"""
        yield from self._tee()

    _len_cache: ClassVar[Dict[int, int]] = {}

    def __len__(self) -> int:
        """返回当前的迭代器的长度，如果无限的话，则回返回最大操作次数。"""
        if isinstance(self.__root__, Hashable):
            hash_value = hash(self.__root__)
            if hash_value in self.__class__._len_cache:
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

    def __mul__(self, times: Union[int, float, str]) -> Self:
        """实现乘法操作"""
        if isinstance(times, SupportsIndex):
            times = int(float(times))
            if times <= 0:
                raise ValueError(f"'times' cannot be negative: {times}")
        try:
            return self.__class__(
                chain.from_iterable(repeat(tuple(self._tee()), int(times)))
            )
        except Exception:
            raise TypeError(f"Unsupported Type: {type(times)}.")

    def __neg__(self) -> Self:
        """定义取负操作"""
        return self.__reversed__()

    def __reversed__(self) -> Self:
        """定义反转"""
        if isinstance(self.__root__, Reversible):
            from copy import deepcopy as copy
            return self.__class__(reversed(copy(self.__root__)))
        else:
            return self.__class__(reversed(list(self._max_gen())))

    def __rshift__(
            self,
            target: Union[Callable[[Iterable[T], Any], R], Sequence]
    ) -> R:
        """实现右移位运算符 >>"""
        if isinstance(target, type) or callable(target):
            result: R = self.collect(target)
        elif isinstance(target, Sequence):
            # noinspection PyArgumentList,PyProtectedMember
            result: R = target.__class__(self.__radd__(target)._tee())
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

    if sys.version_info >= (3, 8):
        def accumulate(
                self, func: Optional[Callable[[T, T], R]] = operator.add, *,
                initial: Optional[int] = None
        ) -> Self:
            return self.__class__(
                accumulate(self._tee(), func, initial=initial)
            )
    else:
        def accumulate(
                self, func: Optional[Callable[[T, T], R]] = operator.add
        ) -> Self:
            return self.__class__(accumulate(self._tee(), func))

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

    def append(self, obj: E) -> Self:
        """将对象附加到 ArkoWrapper 的末尾。注：此方法会修改本身的 __root__"""

        def clean() -> List[Union[E, T]]:
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

    def chain(self, *iterables: Iterable[E]) -> Self:
        """创建一个迭代器，它首先返回第一个可迭代对象中所有元素，接着返回下一个可迭代对象中所有元素，直到耗尽所有可迭代对象中的元素。"""
        return self.__class__(chain(self._tee(), *iterables))

    def collect(
            self,
            func: Optional[Callable[[Iterable[T], Any], R]] = list,
            *args, **kwargs
    ) -> R:
        """以整个迭代器作为参数，于给定的 func 函数中进行运算"""
        return func(self._tee(), *args, **kwargs)

    def combinations(self, r: int = 2) -> Self:
        """返回由元素组成长度为 r 的子序列"""
        return self.__class__(combinations(self._tee(), r))

    def combinations_with_replacement(self, r: int = 2) -> Self:
        """返回由元素组成的长度为 r 的子序列，允许每个元素可重复出现。"""
        return self.__class__(combinations_with_replacement(self._tee(), r))

    def compress(self, selectors: Iterable) -> Self:
        """返回元素中经 selectors 真值测试为 True 的元素"""
        return self.__class__(compress(self._tee(), selectors))

    def cycle(self) -> Self:
        return self.__class__(cycle(self._tee()))

    def drop_while(self, func: Callable[[T], bool]) -> Self:
        return self.__class__(dropwhile(func, self._tee()))

    def empty(self) -> bool:
        return self.__len__() == 0

    def enumerate(self) -> Self:
        def generator() -> Iterator[Tuple[int, T]]:
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

    def extend(self, iterable: Iterable[E]) -> Self:
        if not isinstance(iterable, Iterable):
            raise TypeError(f"{type(iterable)} object is not iterable")

        def generator() -> Iterator[Union[T, E]]:
            yield from self._tee()
            yield from iterable

        return self.__class__(generator())

    def fill(
            self,
            num: int,
            factory: Union[R, Callable[[], R]] = None, *args, **kwargs
    ) -> Self:
        if num > self._max:
            raise ValueError("'num' cannot exceed the maximum number of '_max'")
        elif num < 0:
            raise ValueError("'num' must be a positive number.")

        def generator() -> Iterator[Union[T, R]]:
            yield from self._tee()
            for _ in range(num):
                yield factory(*args, **kwargs) if callable(factory) else factory

        return self.__class__(generator())

    def fill_to(
            self,
            num: int,
            factory: Union[R, Callable[[], R]] = None, *args, **kwargs
    ) -> Self:
        if num < self.length:
            raise ValueError("'num' cannot be less than its own length.")
        length = self.length
        if num == length:
            return self
        else:
            return self.fill(num - length, factory=factory, *args, **kwargs)

    def filter(self, func: Callable[[T], Any]) -> Self:
        return self.__class__(filter(func, self._tee()))

    def filter_false(self, func: Callable[[T], Any]) -> Self:
        return self.__class__(filterfalse(func, self._tee()))

    def find(
            self, func: Callable[[T], bool], *, full: bool = False
    ) -> Iterator[T]:
        for t in self._tee():
            if func(t):
                yield t
                if not full:
                    break

    def find_target(self, target: Any, *, full: bool = False) -> Iterator[int]:
        for t in self.enumerate():
            if t[1] == target:
                yield t[0]
                if not full:
                    break

    def flat(self, depth: int = NOT_SET, *, flat_str: bool = False) -> Self:
        def generator(iterator: Iterable, times: int = NOT_SET):
            for item in iterator:
                if isinstance(item, str) and times:
                    if flat_str and len(item) != 0:
                        yield from item
                    else:
                        yield item
                elif isinstance(item, Iterable) and times:
                    yield from generator(
                        item, (times - 1) if isinstance(times, int) else times
                    )
                else:
                    yield item

        return self.__class__(generator(self._tee(), depth))

    def group(self, n: int, fill_value: Any = NOT_SET) -> Self:
        if not n:
            raise ValueError(f"\'n\' must be a positive integer, not \'{n}\'")

        def generator() -> Iterator[Self]:
            iter_value = iter(self._tee())
            stopped = False
            next_value = NOT_SET
            while not stopped:
                item = self.__class__()
                for _ in range(n):
                    try:
                        if next_value is not NOT_SET:
                            item.append(next_value)
                        else:
                            item.append(next(iter_value))
                    except StopIteration:
                        stopped = True
                        if fill_value is not NOT_SET:
                            item.append(fill_value)
                    try:
                        next_value = next(iter_value)
                    except StopIteration:
                        stopped = True
                        next_value = NOT_SET
                yield item

        return self.__class__(generator())

    # noinspection SpellCheckingInspection
    @overload
    def groupby(self, key: Callable[[T], R] = None) -> Self:
        pass

    # noinspection SpellCheckingInspection
    @overload
    def groupby(
            self,
            start: Callable[[T], bool],
            stop: Callable[[T], bool],
            group_type: Callable[[list], Iterable] = list,
            retain: bool = False
    ) -> Self:
        pass

    # noinspection SpellCheckingInspection
    def groupby(self, *args, **kwargs) -> Self:
        if len(args) + len(kwargs) <= 1:
            return self.__class__(groupby(self._tee(), *args, **kwargs))
        else:
            def make_group(
                    start: Callable[[T], bool],
                    stop: Callable[[T], bool],
                    group_type: Callable[[list], Iterable] = list,
                    retain: bool = False,
                    contain_head: bool = False,
                    contain_tail: bool = False
            ) -> Self:
                def generator() -> Iterator[Iterable[T]]:
                    elem = []
                    other = []
                    flag = False
                    for item in self._tee():
                        if start(item) and not flag:
                            elem.append(item) if contain_head else ...
                            flag = True
                        elif stop(item) and flag:
                            elem.append(item) if contain_tail else ...
                            yield group_type(elem)
                            elem.clear()
                            flag = False
                        elif flag:
                            elem.append(item)
                        elif retain:
                            other.append(item)
                    if elem:
                        yield group_type(elem)
                    if other:
                        yield group_type(other)

                return self.__class__(generator())

            return make_group(*args, **kwargs)

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
            self, func: Callable[[T], R], start: Optional[int] = 0
    ) -> Self:
        def generator() -> Iterator[T]:
            iter_values = iter(self._tee())
            for _ in range(start):
                next(iter_values)
            yield from iter_values

        return self.__class__(map(func, generator()))

    def mutate(
            self, func: Callable[[Iterable[T], Any], Iterable] = list,
            *args, **kwargs
    ) -> Self:
        """改变 __root__ 的类型。"""
        self.__root__ = func(self.__root__, *args, **kwargs)
        return self

    def print(
            self,
            length: Optional[int] = None, *,
            end: Optional[str] = ', ',
            print_func: Optional[Callable[..., Any]] = print
    ) -> Self:
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

    def remove(self, target: Any, *, full: bool = False) -> Self:
        def generator() -> Iterator[T]:
            iter_values = iter(self._tee())
            is_sequence = isinstance(target, Sequence)
            removed = False
            try:
                for _ in range(self.max_operate_time):
                    value = next(iter_values)
                    if (
                            not removed
                            and
                            (
                                    (is_sequence and value in target)
                                    or
                                    value == target
                            )
                    ):
                        removed = full
                    else:
                        yield value
            except StopIteration:
                ...

        return self.__class__(generator())

    def repeat(
            self, times: Optional[Union[int, float, str]] = None
    ) -> Self:
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
            time = int(times)
            if time <= 0:
                raise ValueError(f"'times' cannot be negative: {times}")
            return self.__class__(
                chain.from_iterable(repeat(tuple(self._tee()), time + 1))
            )
        else:
            raise TypeError(f"Unsupported Type: {type(times)}.")

    def reverse(self) -> Self:
        return self.__reversed__()

    @overload
    def slice(self, stop: int) -> Self:  # pragma: no cover
        ...

    @overload
    def slice(
            self, start: int, stop: int, step: Optional[int] = 1
    ) -> Self:  # pragma: no cover
        ...

    def slice(self, *args, **kwargs) -> Self:
        return self.__class__(islice(self._tee(), *args, *kwargs.values()))

    def _iter_integer(self, start: int = 0) -> Iterator[int]:
        iter_values = iter(self._tee())
        for _ in range(start):
            next(iter_values)
        index = start
        for _ in range(self.max_operate_time):
            next(iter_values)
            yield index
            index += 1

    def search(
            self,
            sub: Searchable[E], *,
            func: Callable[[T, E], bool] = operator.eq
    ) -> Iterator[int]:
        target = self.tee()
        sub: ArkoWrapper[E] = ArkoWrapper(sub)
        partial: List[int] = [0]
        for i in sub._iter_integer(1):
            j = partial[i - 1]
            while j > 0 and sub[j] != sub[i]:
                j = partial[j - 1]
            partial.append(j + 1 if sub[j] == sub[i] else j)

        j = 0

        # noinspection PyProtectedMember
        for i in target._iter_integer():
            while j > 0 and target[i] != sub[j]:
                j = partial[j - 1]
            if func(target[i], sub[j]):
                j += 1
            if j == len(sub):
                yield i - j + 1
                j = partial[j - 1]

    def sort(
            self, key: Optional[Callable] = None, reverse: bool = False
    ) -> Self:
        return self.__class__(sorted(self._tee(), key=key, reverse=reverse))

    def starmap(self, func: Callable[[T, T], R]) -> Self:
        return self.__class__(starmap(func, self._tee()))

    def take_while(
            self, func: Union[Any, Callable[[T], bool]] = True
    ) -> Self:
        if callable(func):
            return self.__class__(takewhile(func, self._tee()))
        elif bool(func):
            return self.__deepcopy__()
        else:
            return self.__class__()

    def tee(
            self, n: Optional[int] = None
    ) -> Union[Self, Iterable[Self]]:
        if n is None:
            return self.__class__(self._tee())
        return (self.__class__(item) for item in tee(self.__root__, n))

    def unique(self, key: Optional[Callable] = None) -> Self:
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
                self, *iterables: Iterable[E],
                strict: Optional[bool] = False
        ) -> Self:
            return self.__class__(zip(self._tee(), *iterables, strict=strict))
    else:
        def zip(self, *iterables: Iterable[E]) -> Self:
            return self.__class__(zip(self._tee(), *iterables))

    def zip_longest(
            self, *iterables: Iterable[E],
            fill_value: Optional[R] = None
    ) -> Self:
        return self.__class__(
            zip_longest(
                self.tee(), *ArkoWrapper(iterables).tee(), fillvalue=fill_value
            )
        )
