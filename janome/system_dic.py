import threading
from .sysdic import entries, mmap_entries, connections, chardef, unknowns
from .dic import RAMDictionary, MMapDictionary, UnknownsDictionary


class SystemDictionary(RAMDictionary, UnknownsDictionary):
    __INSTANCE = None
    __lock = threading.Lock()

    @classmethod
    def instance(cls):
        if not cls.__INSTANCE:
            with cls.__lock:
                if not cls.__INSTANCE:
                    cls.__INSTANCE = SystemDictionary(entries(), connections, chardef.DATA, unknowns.DATA)
        return cls.__INSTANCE

    def __init__(self, entries, connections, chardefs, unknowns):
        RAMDictionary.__init__(self, entries, connections)
        UnknownsDictionary.__init__(self, chardefs, unknowns)


class MMapSystemDictionary(MMapDictionary, UnknownsDictionary):
    __INSTANCE = None
    __lock = threading.Lock()

    @classmethod
    def instance(cls):
        if not cls.__INSTANCE:
            with cls.__lock:
                if not cls.__INSTANCE:
                    cls.__INSTANCE = MMapSystemDictionary(mmap_entries(), connections, chardef.DATA, unknowns.DATA)
        return cls.__INSTANCE

    def __init__(self, mmap_entries, connections, chardefs, unknowns):
        MMapDictionary.__init__(self, mmap_entries[0], mmap_entries[1], mmap_entries[2], connections)
        UnknownsDictionary.__init__(self, chardefs, unknowns)
