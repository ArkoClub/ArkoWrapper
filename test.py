from random import randint

import pytest

from arkowrapper import ArkoWrapper

try:
    # noinspection PyPackageRequirements
    from rich.console import Console

    console = Console(color_system='truecolor')
    # noinspection PyShadowingBuiltins
    print = console.print
except ImportError:  # pragma: no cover
    import builtins

    Console = None
    # noinspection PyShadowingBuiltins
    print = builtins.print

data = range(1, 10)
data_a = range(1, 5)
data_b = range(5, 10)
wrapper = ArkoWrapper(data)


@pytest.mark.skip  # pragma: no cover
class TestWrapper(ArkoWrapper):  # pragma: no cover
    __test__ = False


class TestType:  # pragma: no cover
    def test_type_hits(self):
        new_wrapper: TestWrapper[int] = TestWrapper(range(1, 10))
        assert new_wrapper.all()


class TestOperator:  # pragma: no cover
    def test(self):
        from sys import maxsize
        assert ArkoWrapper().__root__ == []
        assert ArkoWrapper(v := randint(0, maxsize)).__root__ == [v]
        with pytest.raises(ValueError):
            ArkoWrapper(max_operate_times=maxsize + 1)
        with pytest.raises(ValueError):
            ArkoWrapper(max_operate_times=-1)
        with pytest.raises(ValueError):
            wrapper.max_operate_time = -5

        print(wrapper.root)
        print(wrapper.max_operate_time)

        test_wrapper = ArkoWrapper(max_operate_times=50)
        test_wrapper.max_operate_time = 10
        assert test_wrapper.max_operate_time == 10

    def test_string(self):
        print()
        print(str(wrapper))
        print(repr(wrapper))

    def test_max_gen(self):
        # noinspection PyTypeChecker
        assert len(list(
            ArkoWrapper(range(100), max_operate_times=20)._max_gen())
        ) == 20
        assert len(list(wrapper._max_gen())) == len(data)

    def test_index(self):
        assert int(ArkoWrapper(range(value := randint(0, 100)))) == value

    def test_len(self):
        from itertools import tee
        assert wrapper.length == len(data)
        assert ArkoWrapper(
            tee(range(100))[0], max_operate_times=10
        ).length == 10

    def test_add(self):
        wrapper_a = ArkoWrapper(data_a)
        wrapper_b = ArkoWrapper(data_b)
        assert list(wrapper_a + wrapper_b) == list(data)
        assert list(data_a + wrapper_b) == list(data)
        assert list(wrapper_a + data_b) == list(data)
        assert list(0 + wrapper) == list(range(10))
        assert list(wrapper + 10) == list(range(1, 11))

    @pytest.mark.flaky(reruns=5)
    def test_equal(self):
        from itertools import tee
        print()
        assert wrapper == ArkoWrapper(data)
        assert wrapper == data
        assert wrapper == list(data)
        assert ArkoWrapper(0) == 0
        assert wrapper + 10 == range(1, 11)
        assert wrapper + 11 != range(1, 11)
        assert wrapper != range(1, 11)
        list_a, list_b = tee(data)
        assert ArkoWrapper(list_a) == list_a

    @pytest.mark.flaky(reruns=3)
    def test_copy(self):
        from copy import (
            copy,
            deepcopy,
        )
        wrapper_a = copy(wrapper)
        wrapper_b = deepcopy(wrapper)
        assert wrapper_a == wrapper_b

    def test_slice(self):
        s = range(30)
        assert ArkoWrapper(s).slice(10) == range(10)
        assert ArkoWrapper(s).slice(10, 20) == range(10, 20)
        assert ArkoWrapper(s).slice(10, 20, 2) == list(range(30))[10:20:2]

    def test_reverse(self):
        assert wrapper.reverse() == data.__reversed__()
        assert reversed(wrapper) == reversed(data)
        assert - wrapper == data.__reversed__()
        assert reversed(ArkoWrapper([1, 2, 3])) == [3, 2, 1]

    def test_getitem(self):
        assert wrapper[5] == data[5]
        assert wrapper @ 5 == data[5]
        assert wrapper[-3] == data[-3]
        assert wrapper[:5] == data[:5]
        assert wrapper[:-2] == data[:-2]
        assert ArkoWrapper(wrapper)[:5] == data[:5]
        with pytest.raises(ValueError):
            print(wrapper[20])
        with pytest.raises(IndexError):
            print(wrapper['a'])

    def test_mul(self):
        assert wrapper * 2 == list(data) + list(data)
        with pytest.raises(ValueError):
            print(wrapper * -1)
        with pytest.raises(TypeError):
            print(wrapper * 'a')

    def test_collect(self):
        assert wrapper.collect() == list(data)
        assert wrapper >> tuple == tuple(data)
        assert wrapper >> tuple([0]) == tuple([0, *data])
        with pytest.raises(TypeError):
            # noinspection PyStatementEffect,PyTypeChecker
            wrapper >> wrapper.root

    def test_accumulate(self):
        import operator
        from itertools import accumulate
        assert wrapper.accumulate() == accumulate(data)
        op = operator.mul
        assert wrapper.accumulate(op) == accumulate(data, op)

    def test_bool(self):
        assert not ArkoWrapper([1, 2, 3, 0, 5]).all()
        assert ArkoWrapper([0, 0, 0, 0, 1, 0]).any()
        assert not ArkoWrapper([0, 0, 0, 0, 0, 0]).any()

    def test_chain(self):
        assert wrapper.chain(range(10, 15), range(15, 20)) == range(1, 20)

    def test_combinations(self):
        ...

    def test_father(self):
        class NewWrapper(ArkoWrapper):
            def test(self):
                return self.root

        new_wrapper = NewWrapper(wrapper)
        new_wrapper.filter_false(lambda: True).filter(lambda: False)


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])
