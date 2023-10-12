import time
import numpy as np
from typing import Generic, TypeVar, List, Tuple, Dict, Callable, Optional, Union

T = TypeVar('T')

class Test(Generic[T]):
    class B():
        def __init__(self, x) -> None:
            self.a: T = x

    def __init__(self, x) -> None:
        super().__init__()
        self.b = self.B(x)

t = Test[bool](1)
t.b.a


# ### todo
# format_name 归属于cluster, filehandle应尽量抽象
# link to cluster 的归属于cluster，filehandle不能操作自身在cluster中的位置，
# cluster对filehandle的操作包括：添加到指定key，（io）移动、删除。