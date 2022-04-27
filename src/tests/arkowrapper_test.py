from arkowrapper import ArkoWrapper
from itertools import groupby


def main():
    wrapper = ArkoWrapper(range(20))
    print(list(iter(wrapper._tee())))
    print(list(wrapper))
    print(list(wrapper.range()))
    print(list(wrapper.range(2)))
    print(list(wrapper.search(list(range(5, 7)))))
    print(list(wrapper.repeat(1).find(18, full=True)))

    print(list(ArkoWrapper('AABBCCDDFFFF').unique()))
    print(wrapper | [1, 3])


if __name__ == '__main__':
    main()
