import queue


class Edge(queue.Queue):
    def __init__(self, starting_point: str, ending_point: str, maxsize=1):
        super().__init__(maxsize=maxsize)

        self.__starting_point = starting_point
        self.__ending_point = ending_point
        self.__name = starting_point + '->' + ending_point

    @property
    def starting_point(self) -> str:
        return self.__starting_point

    @property
    def ending_point(self) -> str:
        return self.__ending_point

    @property
    def name(self) -> str:
        return self.__name
